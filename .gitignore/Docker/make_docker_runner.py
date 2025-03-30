# make_docker_runner.py

bat_code = r"""@echo off
docker run -it --rm ^
  -v "%cd%:/app" ^
  --name noise-analyzer ^
  noise-analyzer
"""

with open("run_docker.bat", "w", encoding="utf-8") as f:
    f.write(bat_code)

print("✅ run_docker.bat 파일이 생성되었어요!")
print("▶ 이걸 더블클릭하면 바로 Docker 컨테이너 실행돼요 🚀")
