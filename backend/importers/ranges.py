"""Seekable, bounded HTTP range reader for Parquet previews of large sources."""
from collections import OrderedDict
import io


class RangeReader(io.RawIOBase):
    def __init__(self,size,fetch,block_size=4*1024*1024):
        super().__init__()
        self.size=size
        self.fetch=fetch
        self.block_size=block_size
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
        if size>128*1024*1024:
            raise OSError('A Parquet sample read exceeds 128 MiB. Use a full streamed import for this source.')
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
