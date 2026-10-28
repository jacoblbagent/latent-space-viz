import re, os, subprocess, pathlib
HERE = pathlib.Path(__file__).resolve().parent.parent / "web"
css = open("/tmp/gf.css").read()
blocks = re.findall(r"/\*\s*([\w\-\[\]]+)\s*\*/\s*(@font-face\s*\{.*?\})", css, re.S)
keep, out_css = [], []
fetched = {}
for sub, block in blocks:
    if sub != "latin":
        continue
    fam = re.search(r"font-family:\s*'([^']+)'", block).group(1)
    wt = re.search(r"font-weight:\s*(\d+)", block).group(1)
    url = re.search(r"url\((https://[^)]+)\)", block).group(1)
    name = f"{fam.replace(' ', '')}-{wt}.woff2"
    fetched.setdefault(url, name)
    b = re.sub(r"url\(https://[^)]+\)", f"url(fonts/{name})", block)
    out_css.append(b)
    keep.append((fam, wt, name))
for url, name in fetched.items():
    dest = HERE / "fonts" / name
    if not dest.exists():
        subprocess.run(["curl", "-sSfL", "--max-time", "30", "-o", str(dest), url], check=True)
(HERE / "fonts.css").write_text("\n".join(out_css))
print("faces:", keep)
print("files:", sorted(p.name + f" {p.stat().st_size//1024}KB" for p in (HERE / "fonts").iterdir()))
