#include "mbed.h"
#include "MQTTNetwork.h"
#include "MQTTmbed.h"
#include "MQTTClient.h"
#include "stm32l475e_iot01_accelero.h"


#include "accelerometer_handler.h"
#include "config.h"
#include "magic_wand_model_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "mbed_rpc.h"
#include "uLCD_4DGL.h"
#include <string.h>
#include <math.h>

#define PI 3.14159265


using namespace std::chrono;

// GLOBAL VARIABLES
WiFiInterface *wifi;
InterruptIn btn2(USER_BUTTON);
//InterruptIn btn3(SW3);
volatile int message_num = 0;
volatile int arrivedcount = 0;
volatile bool closed = false;

const char* topic = "Mbed";

DigitalOut myled1(LED1);
DigitalOut myled2(LED2);
DigitalOut myled3(LED3);

Thread mqtt_thread(osPriorityHigh);
EventQueue mqtt_queue;
uLCD_4DGL uLCD(D1, D0, D2); // serial tx, serial rx, reset pin;

void GestureUI_START(Arguments *in, Reply *out);
RPCFunction rpcGestureUI(&GestureUI_START, "GestureUI_START");

void Tilt_Angle_Detection_START(Arguments *in, Reply *out);
RPCFunction rpcTiltAngleDetection(&Tilt_Angle_Detection_START, "Tilt_Angle_Detection_START");

Thread GestureUI_thread;
Thread Tilt_Angle_Detection_thread;
BufferedSerial pc(USBTX, USBRX);


constexpr int kTensorArenaSize = 60 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
// Return the result of the last prediction

Thread Client_thread;
Config config;

bool MODE = 0;
MQTT::Client<MQTTNetwork, Countdown>* client_out;


int PredictGesture(float* output) {
  // How many times the most recent gesture has been matched in a row
  static int continuous_count = 0;
  // The result of the last prediction
  static int last_predict = -1;

  // Find whichever output has a probability > 0.8 (they sum to 1)
  int this_predict = -1;
  for (int i = 0; i < label_num; i++) {
    if (output[i] > 0.8) this_predict = i;
  }

  // No gesture was detected above the threshold
  if (this_predict == -1) {
    continuous_count = 0;
    last_predict = label_num;
    return label_num;
  }

  if (last_predict == this_predict) {
    continuous_count += 1;
  } else {
    continuous_count = 0;
  }
  last_predict = this_predict;

  // If we haven't yet had enough consecutive matches for this gesture,
  // report a negative result
  if (continuous_count < config.consecutiveInferenceThresholds[this_predict]) {
    return label_num;
  }
  // Otherwise, we've seen a positive result, so clear all our variables
  // and report it
  continuous_count = 0;
  last_predict = -1;

  return this_predict;
}
int angle = 0;
void refresh_uLCD() {
    uLCD.cls(); 
    uLCD.text_width(4); //4X size text
    uLCD.text_height(4);
    for (int i=20; i<=60; i+=20) {
        if(i == angle) {
            uLCD.color(RED);
        } else {
            uLCD.color(GREEN);
        }
        uLCD.printf("%2d\n",i);
    }
}

void publish_confirm_angle_message(MQTT::Client<MQTTNetwork, Countdown>* client) {
    message_num++;
    MQTT::Message message;
    char buff[200];
    sprintf(buff, "%d", angle);
//     sprintf(buff, "QoS0 Hello, Python! #%d", message_num);
    message.qos = MQTT::QOS0;
    message.retained = false;
    message.dup = false;
    message.payload = (void*) buff;
    message.payloadlen = strlen(buff) + 1;
    char angle_topic[] = "confirm_angle";
    int rc = client->publish(angle_topic, message);

    printf("rc:  %d\r\n", rc);
    printf("Puslish message: %s\r\n", buff);
}

int GestureUI() {
        // Whether we should clear the buffer next time we fetch data
    bool should_clear_buffer = false;
    bool got_data = false;

    // The gesture index of the prediction
    int gesture_index;
    config.seq_length = 64;
    config.consecutiveInferenceThresholds[0] = 20;
    config.consecutiveInferenceThresholds[1] = 10;
    config.consecutiveInferenceThresholds[2] = 30;

    config.output_message[0] = "calss0";
    config.output_message[1] = "calss1";
    config.output_message[2] = "calss2";

    myled1.write(1);
    // Set up logging.
    static tflite::MicroErrorReporter micro_error_reporter;
    tflite::ErrorReporter* error_reporter = &micro_error_reporter;

    // Map the model into a usable data structure. This doesn't involve any
    // copying or parsing, it's a very lightweight operation.
    const tflite::Model* model = tflite::GetModel(g_magic_wand_model_data);
    refresh_uLCD();
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        error_reporter->Report(
            "Model provided is schema version %d not equal "
            "to supported version %d.",
            model->version(), TFLITE_SCHEMA_VERSION);
        return -1;
    }

    // Pull in only the operation implementations we need.
    // This relies on a complete list of all the ops needed by this graph.
    // An easier approach is to just use the AllOpsResolver, but this will
    // incur some penalty in code space for op implementations that are not
    // needed by this graph.
    static tflite::MicroOpResolver<6> micro_op_resolver;
    micro_op_resolver.AddBuiltin(
        tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
        tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,
                                tflite::ops::micro::Register_MAX_POOL_2D());
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
                                tflite::ops::micro::Register_CONV_2D());
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
                                tflite::ops::micro::Register_FULLY_CONNECTED());
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
                                tflite::ops::micro::Register_SOFTMAX());
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE,
                                tflite::ops::micro::Register_RESHAPE(), 1);

    // Build an interpreter to run the model with
    static tflite::MicroInterpreter static_interpreter(
        model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
    tflite::MicroInterpreter* interpreter = &static_interpreter;

    // Allocate memory from the tensor_arena for the model's tensors
    interpreter->AllocateTensors();

    // Obtain pointer to the model's input tensor
    TfLiteTensor* model_input = interpreter->input(0);
    if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
        (model_input->dims->data[1] != config.seq_length) ||
        (model_input->dims->data[2] != kChannelNumber) ||
        (model_input->type != kTfLiteFloat32)) {
        error_reporter->Report("Bad input tensor parameters in model");
        return -1;
    }

    int input_length = model_input->bytes / sizeof(float);

    TfLiteStatus setup_status = SetupAccelerometer(error_reporter);
    if (setup_status != kTfLiteOk) {
        error_reporter->Report("Set up failed\n");
        return -1;
    }

    error_reporter->Report("Set up successful...\n");

    while (true) {
        if (MODE)
        {
            myled1.write(0);
            return 0;
        }
        

        // Attempt to read new data from the accelerometer
        got_data = ReadAccelerometer(error_reporter, model_input->data.f,
                                    input_length, should_clear_buffer);

        // If there was no new data,
        // don't try to clear the buffer again and wait until next time
        if (!got_data) {
        should_clear_buffer = false;
        continue;
        }

        // Run inference, and report any error
        TfLiteStatus invoke_status = interpreter->Invoke();
        if (invoke_status != kTfLiteOk) {
        error_reporter->Report("Invoke failed on index: %d\n", begin_index);
        continue;
        }

        // Analyze the results to obtain a prediction
        gesture_index = PredictGesture(interpreter->output(0)->data.f);

        // Clear the buffer next time we read data
        should_clear_buffer = gesture_index < label_num;

        // Produce an output
        if (gesture_index < label_num) {
        // error_reporter->Report(config.output_message[gesture_index]);
            if(gesture_index == 0) {
                angle = 20;
            }
            else if(gesture_index == 1) {
                angle = 40;
            }
            else if(gesture_index == 2) {
                angle = 60;
            }
            refresh_uLCD();
        }
    }
}

void GestureUI_START(Arguments *in, Reply *out) {
    GestureUI_thread.start(GestureUI);
}

// void GestureUI_STOP() {
//     GestureUI_thread.join();
// }
void publish_message() {
    message_num++;
    MQTT::Message message;
    char buff[200];
    // int16_t pDataXYZ[3] = {0};
    // BSP_ACCELERO_AccGetXYZ(pDataXYZ);
    sprintf(buff, "Over_selected_angle");
//     sprintf(buff, "QoS0 Hello, Python! #%d", message_num);
    message.qos = MQTT::QOS0;
    message.retained = false;
    message.dup = false;
    message.payload = (void*) buff;
    message.payloadlen = strlen(buff) + 1;
    int rc = client_out->publish("over", message);

    printf("rc:  %d\r\n", rc);
    printf("Puslish message: %s\r\n", buff);
}

void Tilt_Angle_Detection() {
    myled2.write(1);
    uLCD.cls();
    uLCD.printf("initialization...");
    ThisThread::sleep_for(1000ms);
    BSP_ACCELERO_Init();
    int16_t pDataXYZ[3] = {0};
    int16_t initXYZ[3] = {0};
    double x, y, z;
    double ret = 1;
    BSP_ACCELERO_AccGetXYZ(initXYZ);
    double g = 1.0 * initXYZ[2];
    double out;
    myled2.write(0);
    while (true)
    {
        myled3.write(1);
        if(!MODE) {
            myled3.write(0);
            return;
        }
        BSP_ACCELERO_AccGetXYZ(pDataXYZ);
        x = 1.0*pDataXYZ[0];
        y = 1.0*pDataXYZ[1];
        z = 1.0*pDataXYZ[2];
        ret = z/g;
        out = acos(ret) * 180 / PI;
        uLCD.cls();
        // uLCD.printf("%f\n", x);
        // uLCD.printf("%f\n", y);
        // uLCD.printf("%f\n", z);
        uLCD.text_width(4); //4X size text
        uLCD.text_height(4);
        uLCD.printf("%f\n", out);
        if(out >= angle) {
           publish_message();
        }
        ThisThread::sleep_for(1000ms);
    }    
}

void Tilt_Angle_Detection_START(Arguments *in, Reply *out) {
    Tilt_Angle_Detection_thread.start(Tilt_Angle_Detection);
}

char GestureUI_STOP_MESSAGE[] = "GestureUI_STOP";
char Tilt_Angle_Detection_STOP_MESSAGE[] = "Tilt_Angle_Detection_STOP";
void messageArrived(MQTT::MessageData& md) {
    MQTT::Message &message = md.message;
    char msg[300];
    sprintf(msg, "Message arrived: QoS%d, retained %d, dup %d, packetID %d\r\n", message.qos, message.retained, message.dup, message.id);
    printf(msg);
    ThisThread::sleep_for(1000ms);
    char payload[300];
    sprintf(payload, "%.*s\r\n", message.payloadlen, (char*)message.payload);
    printf(payload);
    // for(int i = 0; i < 14; i++) {
    //     printf("%c %c\n", payload[i], GestureUI_STOP_MESSAGE[i]);
    // }
    // int ret = strncmp(payload, GestureUI_STOP_MESSAGE, 14);
    // printf("%d", ret);
    if(!strncmp(payload, GestureUI_STOP_MESSAGE, 14)) {
        // GestureUI_STOP();
        MODE = 1;
        printf("GestureUI terminated");
    } else if(!strncmp(payload, Tilt_Angle_Detection_STOP_MESSAGE, 20)) {
        MODE = 0;
    }
    ++arrivedcount;
}



void close_mqtt() {
    closed = true;
}

int MQTT_THREAD() {
    wifi = WiFiInterface::get_default_instance();
    if (!wifi) {
            printf("ERROR: No WiFiInterface found.\r\n");
            return -1;
    }


    printf("\nConnecting to %s...\r\n", MBED_CONF_APP_WIFI_SSID);
    int ret = wifi->connect(MBED_CONF_APP_WIFI_SSID, MBED_CONF_APP_WIFI_PASSWORD, NSAPI_SECURITY_WPA_WPA2);
    if (ret != 0) {
            printf("\nConnection error: %d\r\n", ret);
            return -1;
    }


    NetworkInterface* net = wifi;
    MQTTNetwork mqttNetwork(net);
    MQTT::Client<MQTTNetwork, Countdown> client(mqttNetwork);
    client_out = &client;
    //TODO: revise host to your IP
    const char* host = "192.168.43.214";
    printf("Connecting to TCP network...\r\n");

    SocketAddress sockAddr;
    sockAddr.set_ip_address(host);
    sockAddr.set_port(1883);

    printf("address is %s/%d\r\n", (sockAddr.get_ip_address() ? sockAddr.get_ip_address() : "None"),  (sockAddr.get_port() ? sockAddr.get_port() : 0) ); //check setting

    int rc = mqttNetwork.connect(sockAddr);//(host, 1883);
    if (rc != 0) {
            printf("Connection error.");
            return -1;
    }
    printf("Successfully connected!\r\n");

    MQTTPacket_connectData data = MQTTPacket_connectData_initializer;
    data.MQTTVersion = 3;
    data.clientID.cstring = "Mbed";

    if ((rc = client.connect(data)) != 0){
            printf("Fail to connect MQTT\r\n");
    }
    if (client.subscribe(topic, MQTT::QOS0, messageArrived) != 0){
            printf("Fail to subscribe\r\n");
    }
    mqtt_thread.start(callback(&mqtt_queue, &EventQueue::dispatch_forever));
    btn2.rise(mqtt_queue.event(&publish_confirm_angle_message, &client));
    int num = 0;
    while (num != 5) {
            client.yield(100);
            ++num;
    }
    while (1) {
        if (closed) break;
        client.yield(500);
        ThisThread::sleep_for(500ms);
    }
    printf("Ready to close MQTT Network......\n");

    if ((rc = client.unsubscribe(topic)) != 0) {
            printf("Failed: rc from unsubscribe was %d\n", rc);
    }
    if ((rc = client.disconnect()) != 0) {
    printf("Failed: rc from disconnect was %d\n", rc);
    }

    mqttNetwork.disconnect();
    printf("Successfully closed!\n");
    return 0;
}

int main() {
    char buf[256], outbuf[256];
    FILE *devin = fdopen(&pc, "r");
    FILE *devout = fdopen(&pc, "w");
    Client_thread.start(MQTT_THREAD);  
    MODE = 0; 
    while(1) {
        memset(buf, 0, 256);
        for (int i = 0; ; i++) {
            char recv = fgetc(devin);
            if (recv == '\n') {
                printf("\r\n");
                break;
            }
            buf[i] = fputc(recv, devout);
        }
        //Call the static call method on the RPC class
        RPC::call(buf, outbuf);
        printf("%s\r\n", outbuf);
    }
    return 0;
}