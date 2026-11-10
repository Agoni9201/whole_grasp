"""Read the nth MJPEG AVI chunk even when an interrupted writer left no index.

Only complete JPEG chunks are exposed. Offsets come from RIFF chunk boundaries,
never JPEG byte searches or OpenCV's unverified CAP_PROP_POS_FRAMES seeking.
The original recording is opened read-only and is never repaired in place.
"""
from pathlib import Path
import struct


FRAME_READER_VERSION = '20261110-riff1'


class IndexedMJPEGVideo:
    def __init__(self, path):
        self.path = Path(path)
        self.size = self.path.stat().st_size
        self.frames = []
        self.unfinalized = False
        self.truncated_tail = False
        with self.path.open('rb') as stream:
            header = stream.read(12)
            if len(header) != 12 or header[:4] != b'RIFF' or header[8:] != b'AVI ':
                raise ValueError('Expected a stored MJPEG AVI recording')
            self.unfinalized = struct.unpack('<I', header[4:8])[0] == 0xffffffff
            # OpenDML AVI may append RIFF/AVIX containers. Traverse the physical
            # file, while each ordinary LIST still respects its declared end.
            self._scan(stream, 12, self.size, in_video=False)
        if not self.frames:
            raise ValueError(f'No complete MJPEG frames: {self.path}')

    def _scan(self, stream, start, end, *, in_video):
        offset = start
        while offset + 8 <= end:
            stream.seek(offset)
            tag, size = struct.unpack('<4sI', stream.read(8))
            payload = offset + 8
            finish = payload + size
            if tag in (b'LIST', b'RIFF'):
                if size < 4 or payload + 4 > end:
                    raise ValueError('Invalid AVI container')
                kind = stream.read(4)
                if size == 0xffffffff:
                    self.unfinalized = True
                limit = min(finish, end)
                if kind in (b'movi', b'rec ', b'AVIX'):
                    self._scan(stream, payload+4, limit,
                               in_video=in_video or kind == b'movi')
            elif in_video and tag == b'00dc':
                if finish > end:
                    self.truncated_tail = True
                    return
                if size < 4:
                    raise ValueError('Empty AVI video chunk would break frame mapping')
                first = stream.read(2)
                stream.seek(finish-2)
                last = stream.read(2)
                if first != b'\xff\xd8' or last != b'\xff\xd9':
                    raise ValueError('Non-JPEG or damaged AVI frame; cannot verify mapping')
                self.frames.append((payload, size))
            elif in_video and tag.endswith((b'dc', b'db')):
                raise ValueError('Only one MJPEG video stream is supported')
            elif finish > end:
                raise ValueError('Truncated AVI structure outside a video frame')
            offset = finish + size % 2
        if in_video and offset < end:
            self.truncated_tail = True

    def __len__(self):
        return len(self.frames)

    def jpeg(self, index):
        if type(index) is not int or not 0 <= index < len(self):
            raise ValueError('Frame index outside complete recorded video')
        offset, size = self.frames[index]
        with self.path.open('rb') as stream:
            stream.seek(offset)
            content = stream.read(size)
        if len(content) != size or content[:2] != b'\xff\xd8' or content[-2:] != b'\xff\xd9':
            raise ValueError('Stored video changed after its frame index was built')
        return content
