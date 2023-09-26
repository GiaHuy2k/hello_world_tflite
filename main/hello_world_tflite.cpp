#include "sine_model_quantized.hpp"

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_log.h"

// #include "freertos/FreeRTOS.h"
// #include "freertos/task.h"
// #include "esp_log.h"
// #include "driver/gpio.h"

#include "constants.hpp"

#define HIGH 1
#define LOW 0

// gpio_num_t B = GPIO_NUM_5;
// gpio_num_t R = GPIO_NUM_3;
// gpio_num_t G = GPIO_NUM_4;

static const char *TAG = "blink";

namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

constexpr int kTensorArenaSize = 2000;
uint8_t tensor_arena[kTensorArenaSize];
} //namespace


void InitTflite()
{
    MicroPrintf("Start Init Tflite");

    model = tflite::GetModel(sine_model_quantized_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        MicroPrintf("Model provided is schema version %d is not equal to supported "
                    "version %d", model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    static tflite::MicroMutableOpResolver<3> resolver;
    if (resolver.AddQuantize() != kTfLiteOk) {
        return;
    }
    if (resolver.AddFullyConnected() != kTfLiteOk) {
        return;
    }
    if (resolver.AddDequantize() != kTfLiteOk) {
        return;
    }

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk)
    {
        MicroPrintf("allocate_tensor() failed");
        return;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    inference_count = 0;
}

// void InitInternalLED()
// {
//     ESP_LOGI(TAG, "Config GPIO mode");
//     gpio_reset_pin(B);
//     gpio_reset_pin(R);
//     gpio_reset_pin(G);

//     gpio_set_direction(B, GPIO_MODE_OUTPUT);
//     gpio_set_direction(R, GPIO_MODE_OUTPUT);
//     gpio_set_direction(G, GPIO_MODE_OUTPUT);
// }

extern "C" void app_main(void)
{
    InitTflite();

    while (1)
    {
        float position = static_cast<float>(inference_count) /
                        static_cast<float>(kInferencesPerCycle);

        float x = position * kXrange;

        // int8_t x_quantized = x / input->params.scale + input->params.zero_point;

        input->data.f[0] = static_cast<float>(x);
        TfLiteStatus invoke_status = interpreter->Invoke();
        if (invoke_status != kTfLiteOk)
        {
            MicroPrintf("Invoke fail on x: %f\n", static_cast<double>(x));
            return;
        }

        float y = output->data.f[0];
        // float y = (y_quantized - output->params.zero_point) * output->params.scale;
        MicroPrintf("x_value: %f, y_value: \n%f", static_cast<float>(x), static_cast<float>(y));
        // Increment the inference_counter, and reset it if we have reached
        // the total number per cycle
        inference_count += 1;
        if (inference_count >= kInferencesPerCycle) inference_count = 0;
    }
}