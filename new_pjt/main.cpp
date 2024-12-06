#include <iostream>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>


#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "postprocess.h"
#include "rknn_api.h"
#include <dirent.h>
#include "rga.h"
#include "RgaUtils.h"
#include "im2d.h"

using namespace std;

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
           attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
    unsigned char *data;
    int ret;

    data = NULL;

    if (NULL == fp)
    {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0)
    {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char *)malloc(sz);
    if (data == NULL)
    {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}

static unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp;
    unsigned char *data;

    fp = fopen(filename, "rb");
    if (NULL == fp)
    {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}

int detect(char *model_path, char *image_path, char *save_image_path)
{

    rga_buffer_t src;
    rga_buffer_t dst;
    im_rect src_rect;
    im_rect dst_rect;
    memset(&src_rect, 0, sizeof(src_rect));
    memset(&dst_rect, 0, sizeof(dst_rect));
    memset(&src, 0, sizeof(src));
    memset(&dst, 0, sizeof(dst));

    struct timeval start_time, stop_time;

    cout << "image path : " << image_path << endl;
    cv::Mat src_image = cv::imread(image_path, 1);
    if(!src_image.data){
        cout << "image path read fail!" << endl;
        return -1;
    }
    cv::Mat img;
    cv::cvtColor(src_image, img, cv::COLOR_BGR2RGB);

    int img_width = img.cols;
    int img_height = img.rows;

    cout << "RGB img width : " << img_width << " height : " << img_height << endl;

    //todo load model...
    int model_data_size = -1;
    unsigned char* model_data = load_model(model_path, &model_data_size);    //todo

    rknn_context ctx;
    int ret = rknn_init(&ctx, model_data, model_data_size, 0, nullptr);     //todo
    if(ret < 0){
        cout << "rknn model loaded fail! " << endl;
        return -1;
    }

    rknn_sdk_version version;

    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));    //todo

    if(ret < 0){
        cout << "rknn model query version fail! " << endl;
        return -1;
    }

    cout << "sdk version : " << version.api_version << " driver version : " << version.drv_version << endl;

    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));

    if(ret < 0){
        cout << "rknn model query io fail! " << endl;
        return -1;
    }

    cout << "model input num : " << io_num.n_input << " output num : " << io_num.n_output << endl;

    rknn_tensor_attr input_attrs[io_num.n_input];

    memset(input_attrs, 0, sizeof(input_attrs));

    for(int i = 0; i < io_num.n_input; ++i){
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if(ret < 0){
            cout << "rknn model query input attr fial! " << endl;
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for(int i = 0; i < io_num.n_output; ++i){
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        dump_tensor_attr(&(output_attrs[i]));
    }

    int channel = 3;
    int width = 0;
    int height = 0;
    if(input_attrs[0].fmt == RKNN_TENSOR_NCHW){
        cout << "input format is NCHW ! " << endl;
        channel = input_attrs[0].dims[1];
        height = input_attrs[0].dims[2];
        width = input_attrs[0].dims[3];
    }else{
        cout << "input format is NHWC ! " << endl;
        channel = input_attrs[0].dims[3];
        height = input_attrs[0].dims[1];
        width = input_attrs[0].dims[2];
    }

    cout << "input height : " << height << " width : " << width << " channel : " << channel << endl;

    

    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = width * height * channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;

    void* resize_buf = nullptr;

    if(img_width != width || img_height != height){
        cout << "resize with RGA!" << endl;
        resize_buf = malloc(height * width * channel);
        memset(resize_buf, 0x00, height * width *channel);
        src = wrapbuffer_virtualaddr((void*)img.data, img_width, img_height, RK_FORMAT_RGB_888);
        dst = wrapbuffer_virtualaddr((void*)resize_buf, width, height, RK_FORMAT_RGB_888);

        ret = imcheck(src, dst, src_rect, dst_rect);

        if(IM_STATUS_NOERROR != ret){
            cout << "imcheck check error : " << ret << endl;
            return -1;
        }
        IM_STATUS STATUS = imresize(src, dst);
        inputs[0].buf = resize_buf;
    }else{
        inputs[0].buf = (void*)img.data;
    }

    gettimeofday(&start_time, nullptr);
    rknn_inputs_set(ctx, io_num.n_input, inputs);

    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for(int i = 0; i < io_num.n_output; ++i){
        outputs[i].want_float = 0;
    }
    ret = rknn_run(ctx, nullptr);
    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, nullptr);
    gettimeofday(&stop_time, nullptr);

    cout << "model once run use : " << (__get_us(stop_time) - __get_us(start_time)) / 1000 << " ms" << endl;

    //post process
    vector<float> out_scales;
    vector<int32_t> out_zps;
    int8_t* pblob[6];

    for(int i = 0; i < io_num.n_output; ++i){
        out_scales.push_back(output_attrs[i].scale);
        out_zps.push_back(output_attrs[i].zp);
        pblob[i] = (int8_t*) outputs[i].buf;
    }


    GetResultRectYolov8 PostProcess;
    vector<float> DetectiontRects;
    cout << "pblob:" << pblob << endl;
    cout << "out_zps:" << out_zps[0] << endl;
    cout << "out_scales:" << out_scales[0] << endl;


    PostProcess.GetConvDetectionResult(pblob, out_zps, out_scales, DetectiontRects);

    for(int i = 0; i < DetectiontRects.size(); i += 6){
        int classId = int(DetectiontRects[i + 0]);
        float conf = DetectiontRects[i + 1];
        int xmin = int(DetectiontRects[i + 2] * float(img_width) + 0.5);
        int ymin = int(DetectiontRects[i + 3] * float(img_height) + 0.5);
        int xmax = int(DetectiontRects[i + 4] * float(img_width) + 0.5);
        int ymax = int(DetectiontRects[i + 5] * float(img_height) + 0.5);

                char text1[256];
        sprintf(text1, "%d:%.2f", classId, conf);
        rectangle(src_image, cv::Point(xmin, ymin), cv::Point(xmax, ymax), cv::Scalar(255, 0, 0), 2);
        putText(src_image, text1, cv::Point(xmin, ymin + 15), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
    }
    imwrite(save_image_path, src_image);

    printf("== obj: %d \n", int(float(DetectiontRects.size()) / 6.0));

    // release
    ret = rknn_destroy(ctx);

    if (model_data)
    {
        free(model_data);
    }

    if (resize_buf)
    {
        free(resize_buf);
    }

    return 0;
}

int main(){
    char model_path[256] = "./rknn.rknn";
    char image_path[256] = "./test.jpg";
    char save_image_path[256] = "./test_result.jpg";
    detect(model_path, image_path, save_image_path);
    return 0;
}
