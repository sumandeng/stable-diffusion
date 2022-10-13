import argparse
import configparser
import json
import base64
import mimetypes
import os
import ssl
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO

from PIL import Image

from ldm.dream.pngwriter import PngWriter, PromptFormatter
from threading import Event
import pika

tasklock = threading.Lock()

def build_opt(setting, seed, gfpgan_model_exists):
    opt = argparse.Namespace()
    setattr(opt, 'prompt', setting['prompt'])
    setattr(opt, 'init_img', setting['initimg'])
    setattr(opt, 'strength', float(setting['strength']))
    setattr(opt, 'iterations', int(setting['iterations']))
    setattr(opt, 'steps', int(setting['steps']))
    setattr(opt, 'width', int(setting['width']))
    setattr(opt, 'height', int(setting['height']))
    setattr(opt, 'seamless', 'seamless' in setting)
    setattr(opt, 'fit', 'fit' in setting)
    setattr(opt, 'mask', 'mask' in setting)
    setattr(opt, 'invert_mask', 'invert_mask' in setting)
    setattr(opt, 'cfg_scale', float(setting['cfg_scale']))
    setattr(opt, 'sampler_name', setting['sampler_name'])
    setattr(opt, 'gfpgan_strength', float(setting['gfpgan_strength']) if gfpgan_model_exists else 0)
    setattr(opt, 'upscale', [int(setting['upscale_level']), float(setting['upscale_strength'])] if setting[
                                                                                                           'upscale_level'] != '' else None)
    setattr(opt, 'progress_images', 'progress_images' in setting)
    setattr(opt, 'seed', None if int(setting['seed']) == -1 else int(setting['seed']))
    setattr(opt, 'variation_amount', float(setting['variation_amount']) if int(setting['seed']) != -1 else 0)
    setattr(opt, 'with_variations', [])

    broken = False
    if int(setting['seed']) != -1 and setting['with_variations'] != '':
        for part in setting['with_variations'].split(','):
            seed_and_weight = part.split(':')
            if len(seed_and_weight) != 2:
                print(f'could not parse with_variation part "{part}"')
                broken = True
                break
            try:
                seed = int(seed_and_weight[0])
                weight = float(seed_and_weight[1])
            except ValueError:
                print(f'could not parse with_variation part "{part}"')
                broken = True
                break
            opt.with_variations.append([seed, weight])

    if broken:
        raise CanceledException

    if len(opt.with_variations) == 0:
        opt.with_variations = None

    return opt


class CanceledException(Exception):
    pass


class DreamWorker(threading.Thread):
    model = None
    output_dir = None
    canceled = Event()
    result = ''

    def load_config(self):
        config = configparser.ConfigParser()
        config.read("configs/worker_config.ini")
        return config

    def __init__(self):
        threading.Thread.__init__(self)
        self.user_id = None
        self.task_id = None
        config = self.load_config()
        # amqp_url = config["AMQP_URL"]
        amqp_url = 'amqp://admin:admin@127.0.0.1:5672/sdserver'
        parameters = pika.URLParameters(amqp_url)
        parameters.heartbeat = 0
        self.output_dir = "./outputs"

        self.connection = pika.BlockingConnection(parameters)
        self.channel = self.connection.channel()
        self.channel.basic_qos(prefetch_count=1)
        # qname = task_queue_name + '.' + task_queue_key
        qname = 'queue_worker'
        print("waiting tasks from {}".format(qname))
        self.channel.basic_consume(queue=qname, on_message_callback=self.callback, auto_ack=False)
        threading.Thread(
            target=self.channel.basic_consume(queue=qname, on_message_callback=self.callback, auto_ack=False))

    def callback(self, channel, method, properties, body):
        print("[x] Received from {}, body length {}".format(method.routing_key, len(body)))
        with tasklock:
            print(" [*] task start...")
            task_input = json.loads(body)
            self.user_id = task_input['userId']
            self.task_id = task_input['taskId']

            setting = task_input['settings']

            # for key, value in task_input.items():
            #     print(key)
            #
            # print(setting['cfg_scale'])
            # print(setting['fit'])
            # print(setting['gfpgan_strength'])
            # print(setting['height'])
            # print(setting['width'])
            # # print(task['initimg'])
            # print(setting['iterations'])
            # print(setting['prompt'])
            # print(setting['sampler_name'])
            # print(setting['seed'])
            # print(setting['steps'])
            # print(setting['strength'])
            # print(setting['upscale_level'])
            # print(setting['upscale_strength'])
            self.generate(setting)
            self.channel.basic_ack(method.delivery_tag)

    def generate(self, settings):

        # unfortunately this import can't be at the top level, since that would cause a circular import
        from ldm.gfpgan.gfpgan_tools import gfpgan_model_exists

        # post_data = settings
        # print(post_data)
        print("ready to generate")
        opt = build_opt(settings, self.model.seed, gfpgan_model_exists)

        self.canceled.clear()
        print(f">> Request to generate with prompt: {opt.prompt}")
        # In order to handle upscaled images, the PngWriter needs to maintain state
        # across images generated by each call to prompt2img(), so we define it in
        # the outer scope of image_done()
        # config = post_data.copy()  # Shallow copy
        config = settings.copy()
        config['initimg'] = config.pop('initimg_name', '')

        images_generated = 0  # helps keep track of when upscaling is started
        images_upscaled = 0  # helps keep track of when upscaling is completed
        pngwriter = PngWriter(self.output_dir)

        prefix = pngwriter.unique_prefix()

        def publish_image(image: Image, seed, task_id):
            buffer = BytesIO()
            image.save(buffer, format="JPEG")
            image.save("/Users/suman/Downloads/ai2.jpg", format="JPEG")
            exchange = "exchange.notice"
            # result['userId'] = self.user_id
            result = {
                'userId': self.user_id,
                'taskId': task_id,
                'seed': seed,
                'mimeType': 'image/jpeg',
                'data': base64.encodebytes(buffer.getvalue()).decode('utf-8'),
                'args': '',
                'error': ''

            }
            data: str = json.dumps(result)
            # print(data)
            self.channel.basic_publish(exchange=exchange, routing_key="result", body=data.encode('utf-8'))

        # if upscaling is requested, then this will be called twice, once when
        # the images are first generated, and then again when after upscaling
        # is complete. The upscaling replaces the original file, so the second
        # entry should not be inserted into the image list.
        def image_done(image, seed, upscaled=False):
            print("image done.")
            print("生成图片完毕")

            publish_image(image, seed, self.task_id)

            # name = f'{prefix}.{seed}.png'
            # iter_opt = argparse.Namespace(**vars(opt))  # copy
            # if opt.variation_amount > 0:
            #     this_variation = [[seed, opt.variation_amount]]
            #     if opt.with_variations is None:
            #         iter_opt.with_variations = this_variation
            #     else:
            #         iter_opt.with_variations = opt.with_variations + this_variation
            #     iter_opt.variation_amount = 0
            # elif opt.with_variations is None:
            #     iter_opt.seed = seed
            # normalized_prompt = PromptFormatter(self.model, iter_opt).normalize_prompt()
            # path = pngwriter.save_image_and_prompt_to_png(image, f'{normalized_prompt} -S{iter_opt.seed}', name)
            #
            # if int(config['seed']) == -1:
            #     config['seed'] = seed
            # # Append post_data to log, but only once!
            # if not upscaled:
            #     with open(os.path.join(self.output_dir, "dream_web_log.txt"), "a") as log:
            #         log.write(f"{path}: {json.dumps(config)}\n")
            #
            #     # self.wfile.write(bytes(json.dumps(
            #     #     {'event': 'result', 'url': path, 'seed': seed, 'config': config}
            #     # ) + '\n', "utf-8"))
            #     self.result = json.dumps(
            #         {'event': 'result', 'url': path, 'seed': seed, 'config': config}
            #     )
            #
            # # control state of the "postprocessing..." message
            # upscaling_requested = opt.upscale or opt.gfpgan_strength > 0
            # nonlocal images_generated  # NB: Is this bad python style? It is typical usage in a perl closure.
            # nonlocal images_upscaled  # NB: Is this bad python style? It is typical usage in a perl closure.
            # if upscaled:
            #     images_upscaled += 1
            # else:
            #     images_generated += 1
            # if upscaling_requested:
            #     action = None
            #     if images_generated >= opt.iterations:
            #         if images_upscaled < opt.iterations:
            #             action = 'upscaling-started'
            #         else:
            #             action = 'upscaling-done'
            #     if action:
            #         x = images_upscaled + 1
            #         # self.wfile.write(bytes(json.dumps(
            #         #     {'event': action, 'processed_file_cnt': f'{x}/{opt.iterations}'}
            #         # ) + '\n', "utf-8"))
            #         self.result = json.dumps(
            #             {'event': action, 'processed_file_cnt': f'{x}/{opt.iterations}'}
            #         )

        step_writer = PngWriter(os.path.join(self.output_dir, "intermediates"))
        step_index = 1

        def image_progress(sample, step):
            if self.canceled.is_set():
                # self.wfile.write(bytes(json.dumps({'event': 'canceled'}) + '\n', 'utf-8'))
                self.result = json.dumps({'event': 'canceled'})
                raise CanceledException
            path = None
            # since rendering images is moderately expensive, only render every 5th image
            # and don't bother with the last one, since it'll render anyway
            nonlocal step_index
            if opt.progress_images and step % 5 == 0 and step < opt.steps - 1:
                image = self.model.sample_to_image(sample)
                name = f'{prefix}.{opt.seed}.{step_index}.png'
                metadata = f'{opt.prompt} -S{opt.seed} [intermediate]'
                path = step_writer.save_image_and_prompt_to_png(image, metadata, name)
                step_index += 1
            # self.wfile.write(bytes(json.dumps(
            #     {'event': 'step', 'step': step + 1, 'url': path}
            # ) + '\n', "utf-8"))
            self.result = json.dumps(
                {'event': 'step', 'step': step + 1, 'url': path}
            )

        try:
            if opt.init_img is None:
                # Run txt2img
                self.model.prompt2image(**vars(opt), step_callback=image_progress, image_callback=image_done)
            else:
                # Decode initimg as base64 to temp file
                with open("./img2img-tmp.png", "wb") as f:
                    initimg = opt.init_img.split(",")[1]  # Ignore mime type
                    f.write(base64.b64decode(initimg))
                opt1 = argparse.Namespace(**vars(opt))
                opt1.init_img = "./img2img-tmp.png"

                try:
                    # Run img2img
                    self.model.prompt2image(**vars(opt1), step_callback=image_progress, image_callback=image_done)
                finally:
                    # Remove the temp file
                    os.remove("./img2img-tmp.png")
        except CanceledException:
            print(f"Canceled.")
            return

    def run(self):
        print("start thread:", self.native_id)
        self.channel.start_consuming()

# class ThreadingDreamWorker(ThreadingHTTPServer):
#     def __init__(self, server_address):
#         super(ThreadingDreamWorker, self).__init__(server_address, DreamWorker)
