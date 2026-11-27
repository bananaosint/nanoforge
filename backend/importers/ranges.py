"""Seekable, bounded HTTP range reader for remote Parquet."""
from collections import OrderedDict
import io
import time


class RangeReader(io.RawIOBase):
    def __init__(self,size,fetch,block_size=4*1024*1024,*,max_read_size=128*1024*1024):
        super().__init__()
        self.size=size
        self.fetch=fetch
        self.block_size=block_size
        self.max_read_size=max_read_size
        self.position=0
        self.blocks=OrderedDict()

    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position

    def seek(self,offset,whence=0):
        position=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        if position<0:raise ValueError('Negative range seek.')
        self.position=position
        return position

    def read(self,size=-1):
        size=max(0,self.size-self.position) if size<0 else min(size,max(0,self.size-self.position))
        if size>self.max_read_size:
            raise OSError(f'A Parquet range read exceeds {self.max_read_size // (1024 * 1024)} MiB.')
        output=[]
        while size:
            index=self.position//self.block_size
            if index not in self.blocks:
                start=index*self.block_size
                end=min(self.size,start+self.block_size)-1
                data=self.fetch(start,end)
                if len(data)!=end-start+1:raise OSError('Incomplete HTTP byte range.')
                self.blocks[index]=data
                while len(self.blocks)>2:self.blocks.popitem(last=False)
            data=self.blocks[index]
            self.blocks.move_to_end(index)
            offset=self.position-index*self.block_size
            chunk=data[offset:offset+size]
            if not chunk:break
            output.append(chunk);self.position+=len(chunk);size-=len(chunk)
        return b''.join(output)

    def readinto(self,buffer):
        data=self.read(len(buffer));buffer[:len(data)]=data
        return len(data)


def remote_file(shard, request, open_url, cancelled, timeout=15, *, max_read_size=128*1024*1024):
    """Share range validation between sampling and training."""
    def fetch(start, end):
        if cancelled():
            raise InterruptedError()
        url = shard['url']
        url += ('&' if '?' in url else '?') + 'nanoforge_range=' + str(time.time_ns())
        headers = {'Range': f'bytes={start}-{end}'}
        with open_url(request(url, headers), timeout=timeout) as response:
            if response.status == 206:
                expected = f"bytes {start}-{end}/{shard['size']}"
                if response.headers.get('Content-Range') != expected:
                    raise OSError('Server returned the wrong Parquet byte range.')
                return response.read(end - start + 2)
            if response.status == 200 and shard['size'] <= 4 * 1024 * 1024:
                data = response.read(shard['size'] + 1)
                return data[start:end + 1]
            raise OSError('Server ignored HTTP range requests. Use Local cache for this dataset.')
    return RangeReader(shard['size'], fetch, max_read_size=max_read_size)
