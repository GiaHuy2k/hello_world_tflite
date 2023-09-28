# Hướng dẫn gen thư viện chạy ML 
- Vào [link](https://github.com/tensorflow/tflite-micro) này và clone repo về
- Các bạn cài python và tạo virtual environment theo [link](https://docs.python.org/3/library/venv.html)
- Sau đó activate virtual env và cài các thư viện cần thiết: numpy, matplotlib, pillow (mình nhớ nhiêu đó thôi ^.^)
- Các bạn có thể đọc file readme (tensorflow/lite/micro/docs/new_platform_support.md) trong repo và làm theo bước 1 để gen ra folder ```/tmp/tflm-tree```
- Vô folder ```/tmp/tflm-tree``` và copy ba folder ```signal```, ```tensorflow```, ```third_party``` vào trong prj các bạn và tạo CMakeLists.txt để build.