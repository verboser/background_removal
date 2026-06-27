import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/oxford_pets/images.tar.gz"
FILE_LIST = ROOT / "ml/oxford_pets_eval50.txt"
OUT_DIR = ROOT / "data/oxford_pets/images_eval50"


if not ARCHIVE.exists():
    raise SystemExit(f"Архив с картинками не найден: {ARCHIVE}")
if not FILE_LIST.exists():
    raise SystemExit(f"Список eval50 не найден: {FILE_LIST}")

names = []
for line in FILE_LIST.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#"):
        names.append(line)

OUT_DIR.mkdir(parents=True, exist_ok=True)
needed = {f"images/{name}.jpg" for name in names}

with tarfile.open(ARCHIVE, "r:gz") as tar:
    members = [member for member in tar.getmembers() if member.name in needed]
    found = {member.name for member in members}
    missing = sorted(needed - found)
    if missing:
        raise SystemExit("В архиве не найдены файлы: " + ", ".join(missing[:10]))

    for member in members:
        member.name = Path(member.name).name
        tar.extract(member, OUT_DIR, filter="data")

print(f"Извлечено {len(members)} файлов в {OUT_DIR}")
