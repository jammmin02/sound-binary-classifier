@echo off
docker run -it --rm ^
  -v "%cd%:/app" ^
  --name noise-analyzer ^
  noise-analyzer
