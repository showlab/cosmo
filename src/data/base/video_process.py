import decord
from decord import cpu, gpu
import ffmpeg
import numpy as np
import random
import torch
from subprocess import Popen, PIPE
import tempfile
import pytorchvideo.transforms as video_transforms
import imageio
import cv2
try:
    from azfuse import File
except Exception as e:
    print("azfuse not supported on this cluster, use local file system instead")

FFMPEG_RAW_VIDEO_FRAME_SIZE = 360

def video_augment(video_frame=3, video_image_size=224, mode='train', aug_type='randaug'):
    # Create a video clip transform sequence
    # FROM [C,T,H,W] TO [C,T,H,W]
    # https://pytorchvideo.readthedocs.io/en/latest/api/transforms/transforms.html
    return video_transforms.create_video_transform(mode=mode, 
                                                    num_samples=video_frame,
                                                    min_size=int(video_image_size*1.2),
                                                    max_size=int(video_image_size*1.5),
                                                    crop_size=video_image_size,
                                                    aug_type=aug_type,  # randaug/augmix
                                                    )

def sample_frames(num_frames, vlen, sample='rand', fix_start=None):
        acc_samples = min(num_frames, vlen)
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
        elif fix_start is not None:
            frame_idxs = [x[0] + fix_start for x in ranges]
        elif sample == 'uniform':
            frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        return frame_idxs
    
def read_frames_decord_from_path(video_path, num_frames, read_video_by_azfuse=False, mode='train', fix_start=None):
    """
    frames: T, H, W, C
    """
    # self.custom_logger.info("video path: {}".format(video_path))
    if mode in ['train', 'val']:
        sample = 'rand'
    else:
        sample = 'uniform'
    # video_reader = decord.VideoReader(video_path, width=512, height=512, num_threads=1, ctx=cpu(0))
    # can use gpu for speed up reading
    # video_reader = decord.VideoReader(video_path, width=256, height=256, num_threads=1, ctx=cpu(0))
    if read_video_by_azfuse:
        with File.open(video_path, 'rb') as f:
            video_reader = decord.VideoReader(f, width=256, height=256, num_threads=1, ctx=cpu(0))
    else:
        with open(video_path, 'rb') as f:
            video_reader = decord.VideoReader(f, width=256, height=256, num_threads=1, ctx=cpu(0))
    decord.bridge.set_bridge('torch')
    vlen = len(video_reader)
    frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
    frames = video_reader.get_batch(frame_idxs).byte()
    frames = frames #to  T, C, H, W
    return frames, frame_idxs, vlen

def read_frames_decord(video_reader, num_frames, mode='train', fix_start=None):
    sample = 'rand' if mode in ['train', 'val'] else 'uniform'
    
    decord.bridge.set_bridge('torch')
    vlen = len(video_reader)
    frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
    frames = video_reader.get_batch(frame_idxs).byte()
    return frames, frame_idxs, vlen


def read_frames_from_timestamps_and_path(video_reader, num_frames, times_array):
    """
    return c,t,h,w
    """
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    decord.bridge.set_bridge('torch')
    frame_idxs_array = []
    for beg_time, end_time in times_array:   
        frame_idxs = sample_frames_from_seq(num_frames, vlen, sample='rand', begin_index=int(beg_time * fps), end_index=int(end_time * fps), fix_start=None)
        frame_idxs_array.extend(frame_idxs)
    # frame_idxs = sample_frames_from_seq(num_frames, vlen, sample='rand', begin_index=int(beg_time * fps), end_index=int(end_time * fps), fix_start=None)
    frames = video_reader.get_batch(frame_idxs_array).byte() # t, h, w, c
    frames = frames.permute(3, 0, 1, 2)
    return frames

def sample_frames_from_seq(num_frames, vlen, sample='rand', begin_index=0, end_index=1., fix_start=None):
    """
    The decord seq idxs start from 0 to vlen-1
    """

    distance = end_index - begin_index
    # Ensure that begin_index and end_index are within valid bounds
    if begin_index < 0:
        begin_index = 0
        end_index += distance
    if end_index >= vlen:
        end_index = vlen - 1
        begin_index = end_index - distance

    if end_index < begin_index:
        print(f"Begin: {begin_index}, end: {end_index}")
        raise ValueError("Begin index larger than end index.")

    intervals = np.linspace(start=begin_index, stop=end_index, num=num_frames + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
    if sample == 'rand':
        frame_idxs = []
        for x in ranges:
            if x[1] > x[0]:
                frame_idxs.append(random.choice(range(x[0], x[1])))
            else:
                frame_idxs.append(x[0])
    elif fix_start is not None:
        frame_idxs = [x[0] + fix_start for x in ranges]
    elif sample == 'uniform':
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
    else:
        raise NotImplementedError
    # sort these idxs
    frame_idxs = sorted(frame_idxs)
    return frame_idxs



def read_frames_from_timestamps_ffmpeg(video_path, num_frames, mode='train', start=0., end=1., read_video_by_azfuse=False):
    """
    return: c t h w
    for example, start: 11.0, end: 16.0
    call ffmpeg directly for efficient as other packages need to read all frames from start to end
    1. decode desired frames from start to end (10 to 20 frames)
    2. sample num_frames frames from the decoded frames
    thr original video is 640x360, we crop it to 360x360 for further augmentation
    """
    num_sec = end - start
    start_seek = start
    if mode == "train":
        center_crop = False
        random_flip = True
        sample = 'rand'
        crop_only = False
    else:
        center_crop = True
        random_flip = False
        sample = 'uniform'
        crop_only = True
    if center_crop:
        aw, ah = 0.5, 0.5
    else:
        aw, ah = random.uniform(0, 1), random.uniform(0, 1)
    # dynamically calculate fps based on desired number of frames and video length
    desired_frames = random.randint(10, 20)
    fps = desired_frames / num_sec
    if read_video_by_azfuse:
        with File.open(video_path, "rb") as infile:
            temp = tempfile.NamedTemporaryFile()
            temp.write(infile.read())
            temp.flush()
            video_path = temp.name
        
    cmd = (
        ffmpeg
        .input(video_path, ss=start_seek, t=num_sec + 0.01)
        .filter('fps', fps=fps)
    )
    # original: 640 x 360
    if crop_only:
        cmd = (
            cmd.crop('(iw - {})*{}'.format(FFMPEG_RAW_VIDEO_FRAME_SIZE, aw),
                        '(ih - {})*{}'.format(FFMPEG_RAW_VIDEO_FRAME_SIZE, ah),
                        str(FFMPEG_RAW_VIDEO_FRAME_SIZE), str(FFMPEG_RAW_VIDEO_FRAME_SIZE))
        )
    else:
        cmd = (
            cmd.crop('(iw - min(iw,ih))*{}'.format(aw),
                        '(ih - min(iw,ih))*{}'.format(ah),
                        'min(iw,ih)',
                        'min(iw,ih)')
            .filter('scale', FFMPEG_RAW_VIDEO_FRAME_SIZE, FFMPEG_RAW_VIDEO_FRAME_SIZE)
        )
    if random_flip and random.uniform(0, 1) > 0.5:
        cmd = cmd.hflip()
    out, _ = (
        cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24').run(capture_stdout=True, quiet=True) # set False for debug
    )
    video = np.frombuffer(out, np.uint8).reshape([-1, FFMPEG_RAW_VIDEO_FRAME_SIZE, FFMPEG_RAW_VIDEO_FRAME_SIZE, 3])
    video_tensor = torch.from_numpy(np.copy(video))
    video_tensor = (video_tensor.permute(3, 0, 1, 2) + 0.01).type(torch.uint8)  # prevent all dark vide
    vlen = video_tensor.shape[1]
    if vlen < num_frames:
        zeros = torch.ones((3, num_frames - vlen, FFMPEG_RAW_VIDEO_FRAME_SIZE, FFMPEG_RAW_VIDEO_FRAME_SIZE), dtype=torch.uint8)
        video_tensor = torch.cat((video_tensor, zeros), axis=1)
    return video_tensor


def read_frames_gif(video_path, num_frames, mode='train', fix_start=None, resize_shape=(256, 256)):
    """
    read from gif and resize to 256x256
    return: 
    """
    if mode == 'train':
        sample = 'rand'
    else:
        sample = 'uniform'
    gif = imageio.get_reader(video_path)
    vlen = len(gif)
    frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
    # print(video_path)
    frames = []
    for index, frame in enumerate(gif):
        if index in frame_idxs:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            frame = cv2.resize(frame, resize_shape)  # Resize frame to a common shape
            frame = torch.from_numpy(frame).byte()
            frame = frame.permute(2, 0, 1)
            frames.append(frame)
    frames = torch.stack(frames)
    frames = frames.permute(0, 2, 3, 1) # from t, c, h, w to t, h, w, c
    return frames, frame_idxs, vlen