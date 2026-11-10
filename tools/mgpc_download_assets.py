"""Inspect/download only the public pretrained assets linked by MGPC's README."""
import argparse
import json
from pathlib import Path
import re
import requests

FOLDER = "https://drive.google.com/drive/folders/18gf1-w6TE4S1EhxWX1SDkPUxWCi5b_wR"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--probe-range", action="store_true")
    parser.add_argument("--parallel", type=int, default=0)
    parser.add_argument("--probe-start", type=int, default=0)
    parser.add_argument("--probe-size", type=int, default=1024)
    parser.add_argument("--probe-all", action="store_true")
    parser.add_argument("--chunk-mb", type=int, default=16)
    parser.add_argument("--cancel-downloads", action="store_true")
    parser.add_argument("--front-end", choices=("usercontent","drive"),default="usercontent")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.cancel_downloads:
        import os, signal
        for file in Path('/proc').glob('[0-9]*/cmdline'):
            try:
                command=file.read_bytes().decode().split('\0')
                if (len(command)>2 and command[1].endswith('tools/mgpc_download_assets.py')
                        and ('--parallel' in command or '--download' in command)
                        and int(file.parent.name)!=os.getpid()):
                    os.kill(int(file.parent.name),signal.SIGTERM)
                    print('Stopped task downloader',file.parent.name,flush=True)
            except (OSError,UnicodeError): pass
        return
    cache=args.output_dir/'public_folder.html'
    if cache.exists():
        response=requests.Response();response.status_code=200;response._content=cache.read_bytes()
    else:
        response = requests.get(FOLDER, timeout=60)
        response.raise_for_status()
    if args.parallel:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import shutil
        import time
        url = "https://drive.google.com/uc" if args.front_end=="drive" else "https://drive.usercontent.google.com/download"
        params = dict(id="1hF4u9aaQGfjx0PepdktBh8OAFRvWmoVJ", export="download", confirm="t")
        with requests.get(url, params=params, headers={"Range":"bytes=0-0"}, stream=True, timeout=(15,60)) as head:
            if head.status_code != 206: raise RuntimeError("Server does not support validated range downloads")
            total = int(head.headers["Content-Range"].split("/")[-1])
        chunk_size = max(1,args.chunk_mb)*1024*1024
        parts = args.output_dir / f"download_parts_{args.chunk_mb}mb"
        parts.mkdir(exist_ok=True)
        ranges = [(i,start,min(total,start+chunk_size)-1) for i,start in enumerate(range(0,total,chunk_size))]
        # Reuse only complete prefixes already received by the sequential
        # download. Its later append cannot change these earlier bytes.
        sequential = next(args.output_dir.glob("ckpt_8192.pt*.part"), None)
        if sequential:
            complete = sequential.stat().st_size//chunk_size
            with sequential.open("rb") as stream:
                for index in range(complete):
                    path = parts / f"{index:04d}.part"
                    data = stream.read(chunk_size)
                    if len(data) != chunk_size: break
                    if not path.exists(): path.write_bytes(data)
        import threading
        connections=threading.local()
        def fetch(spec):
            index,start,end = spec
            path = parts / f"{index:04d}.part"
            if path.exists() and path.stat().st_size == end-start+1: return index
            if not hasattr(connections,"session"): connections.session=requests.Session()
            for attempt in range(4):
                try:
                    with connections.session.get(url, params=params, headers={"Range":f"bytes={start}-{end}","Accept-Encoding":"identity"},
                                      stream=True, timeout=(15,45)) as reply:
                        reply.raise_for_status()
                        if reply.status_code != 206 or reply.headers.get("Content-Range") != f"bytes {start}-{end}/{total}":
                            raise RuntimeError("Wrong Content-Range; refusing corrupted checkpoint")
                        temporary = path.with_suffix(".incomplete")
                        with temporary.open("wb") as out:
                            for block in reply.iter_content(64*1024): out.write(block)
                        if temporary.stat().st_size != end-start+1: raise RuntimeError("Incomplete model segment")
                        temporary.replace(path)
                    return index
                except Exception as error:
                    print(f"Retry segment={index} attempt={attempt+1} error={type(error).__name__}",flush=True)
                    if attempt == 3: raise
                    time.sleep(1+attempt)
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=max(1,min(args.parallel,16))) as pool:
            jobs = [pool.submit(fetch,spec) for spec in ranges]
            for done,job in enumerate(as_completed(jobs),1):
                job.result()
                if done%8==0 or done==len(jobs): print(f"Segments {done}/{len(jobs)}, elapsed {time.monotonic()-started:.1f}s",flush=True)
        destination = args.output_dir / "ckpt_8192.pt"
        temporary = destination.with_suffix(".assembling")
        with temporary.open("wb") as out:
            for index,_,_ in ranges:
                with (parts / f"{index:04d}.part").open("rb") as stream: shutil.copyfileobj(stream,out)
        if temporary.stat().st_size != total: raise RuntimeError("Checkpoint length mismatch")
        temporary.replace(destination)
        (args.output_dir / "source.json").write_text(json.dumps(dict(folder=FOLDER,
            file_id=params["id"], filename=destination.name, bytes=total, validated_ranges=True),indent=2)+"\n")
        print("Checkpoint ready:",destination,flush=True)
        return
    (args.output_dir / "public_folder.html").write_text(response.text)
    # A bounded excerpt for inspection; public asset metadata only.
    hits = []
    for match in re.finditer(r"[^\"<>\\\n]{0,100}\.(?:pt|pth|bin|safetensors|zip)[^\"<>\\\n]{0,80}", response.text):
        value = match.group(0)
        if value not in hits: hits.append(value)
    print(json.dumps(dict(status=response.status_code, asset_names=hits[:35]), ensure_ascii=False, indent=2))
    if args.probe_range:
        import time
        started=time.monotonic()
        url = "https://drive.google.com/uc" if args.front_end=="drive" else "https://drive.usercontent.google.com/download"
        end = args.probe_start + args.probe_size - 1
        with requests.get(url, params=dict(id="1hF4u9aaQGfjx0PepdktBh8OAFRvWmoVJ", export="download", confirm="t"),
                          headers={"Range": f"bytes={args.probe_start}-{end}"}, stream=True, timeout=(15, 30)) as probe:
            print("RANGE", probe.status_code, {key:probe.headers.get(key) for key in
                ("Content-Length", "Content-Range", "Accept-Ranges", "Content-Type")},flush=True)
            print('Response headers in',round(time.monotonic()-started,2),'s',flush=True)
            if args.probe_all:
                import time
                start=time.monotonic();data=probe.content
                print('Full probe',len(data),'bytes in',round(time.monotonic()-start,2),'s',flush=True)
                return
            print("first block bytes", len(next(probe.iter_content(min(args.probe_size,4096)))) ,flush=True)
    if args.download:
        import gdown
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")
        files = {node["data-id"]: node.get_text() for node in soup.find_all(attrs={"data-id": True})
                 if "ckpt_8192.pt" in node.get_text()}
        if len(files) != 1:
            raise RuntimeError("Cannot uniquely identify official 8192-point checkpoint")
        file_id = next(iter(files))
        destination = args.output_dir / "ckpt_8192.pt"
        if destination.is_file() and destination.stat().st_size > 1_000_000_000:
            print("Checkpoint already downloaded:", destination)
        else:
            result = gdown.download(id=file_id, output=str(destination), use_cookies=False,
                                    resume=True, timeout=(15, 90))
            if result is None: raise RuntimeError("MGPC checkpoint download failed")
        (args.output_dir / "source.json").write_text(json.dumps(dict(
            folder=FOLDER, file_id=file_id, filename=destination.name,
            bytes=destination.stat().st_size), indent=2) + "\n")

if __name__ == "__main__": main()
