"""Read official ZIP metadata over strict HTTP ranges; never decode imagery."""
import io
import json
import urllib.request
import zipfile

URL = "https://zenodo.org/api/records/7223446/files/OpenEarthMap.zip/content"
SIZE = 9099481727


class RemoteArchive(io.RawIOBase):
    def __init__(self, opener=urllib.request.urlopen, *, url=URL, size=SIZE):
        self.position = 0
        self.transferred = 0
        self.opener = opener
        self.url = url
        self.size = size

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset if whence == 2 else -1
        if position < 0 or position > self.size:
            raise ValueError("Invalid archive seek")
        self.position = position
        return position

    def read(self, size=-1):
        size = self.size-self.position if size < 0 else min(size, self.size-self.position)
        if not size:
            return b""
        if size > 16*1024*1024:
            raise ValueError("Refusing oversized range; use bounded reads")
        end = self.position+size-1
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.position}-{end}"})
        with self.opener(request, timeout=30) as response:
            expected = f"bytes {self.position}-{end}/{self.size}"
            if response.status != 206 or response.headers.get("Content-Range") != expected:
                raise RuntimeError("Server did not honor exact range; no payload read")
            content = response.read(size+1)
        if len(content) != size:
            raise RuntimeError("Range length mismatch")
        self.position += size
        self.transferred += size
        return content


def inventory(archive):
    entries = [item for item in archive.infolist() if "christchurch" in item.filename.lower().split("/") and not item.is_dir()]
    return [{"name": item.filename, "compressed_bytes": item.compress_size,
             "bytes": item.file_size, "crc32": f"{item.CRC:08x}",
             "compression": item.compress_type, "header_offset": item.header_offset} for item in entries]


if __name__ == "__main__":
    remote = RemoteArchive()
    with zipfile.ZipFile(remote) as archive:
        selected = inventory(archive)
    print(json.dumps({"archive_bytes": SIZE, "metadata_transferred_bytes": remote.transferred,
                      "region": "christchurch", "entries": selected,
                      "selected_compressed_bytes": sum(row["compressed_bytes"] for row in selected)}, indent=2))
