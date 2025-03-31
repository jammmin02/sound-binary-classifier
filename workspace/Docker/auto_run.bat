echo off
SET project_name=noise-analyzer

echo [1] Docker 이미지 빌드 중...
docker build -t %project_name% .

echo [2] VSCode 열기
code .

echo [3] Docker 컨테이너 실행
docker run -it --rm -v %cd%:/app --name %project_name% %project_name%