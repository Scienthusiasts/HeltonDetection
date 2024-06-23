import onnx
import torch
import os
from torch import nn
import onnxruntime
# onnx-simplifier
import onnxsim

from utils.runnerUtils import *



def torchExportOnnx(model:nn.Module, device:str, input_size:list[int], export_dir:str, export_name:str, export_param:dict, ckpt_path=False, ):
    if not os.path.isdir(export_dir):os.makedirs(export_dir)
    export_path = os.path.join(export_dir, export_name)
    model = model.to('cpu')
    # 导入预训练权重
    if ckpt_path:
        model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    # 在调用torch.onnx.export之前，需要先创建输入数据x
    # 基于追踪（trace）的模型转换方法：给定一组输入，实际执行一遍模型，把这组输入对应的计算图记录下来，保存为 ONNX 格式(静态图)
    x = torch.randn(1, 3, input_size[0], input_size[1]).to('cpu')
    with torch.no_grad():
        torch.onnx.export(
            # 要转换的模型
            model,                   
            # 模型的任意一组输入
            x,                       
            # 导出的 ONNX 文件名
            export_path,             
            # ONNX 算子集版本: https://onnx.ai/onnx/operators/
            opset_version=11,        
            # 将可以在编译时计算的常量表达式提前计算并替换掉相应的运算
            do_constant_folding=True,
            # 输入 Tensor 的名称, 如果不指定，会使用默认名字
            input_names=export_param['input_names'],   
            # 输出 Tensor 的名称, 如果不指定，会使用默认名字
            output_names=export_param['output_names'],  
            # 动态输入输出设置:
            dynamic_axes=export_param['dynamic_axes']
        ) 

    # 读取 ONNX 模型
    onnx_model = onnx.load(export_path)
    # 检查模型格式是否正确
    onnx.checker.check_model(onnx_model)
    print('无报错, onnx模型导出成功')
    # 以可读的形式打印计算图
    # print(onnx.helper.printable_graph(onnx_model.graph))
    # NETRON在线平台可视化模型结构 https://netron.app/
    '''使用onnx-simplifier来进行onnx的简化'''
    # python -m onnxsim [model_path_before_sim].onnx [model_path_after_sim].onnx
    print(f"Simplifying with onnx-simplifier {onnxsim.__version__}...")
    model_onnx, check = onnxsim.simplify(model_onnx)
    assert check, "assert check failed !"
    onnx.save(model_onnx, export_path)






def onnxInferenceSingleImg(model, device, class_names, image2color, img_size, tf, img_path, onnx_path, save_vis_path=None, T=0.3, agnostic=False, show_text=True, vis_heatmap=False):
    '''推理一张图
        # Args:
            - device:        cpu/cuda
            - class_names:   每个类别的名称, list
            - image2color:   每个类别一个颜色
            - img_size:      固定图像大小 如[832, 832]
            - tf:            数据预处理(基于albumentation库)
            - img_path:      图像路径
            - save_vis_path: 可视化图像保存路径
            - ckpt_path:     模型权重路径
            - T:             可视化的IoU阈值

        # Returns:
            - boxes:       网络回归的box坐标    [obj_nums, 4]
            - box_scores:  网络预测的box置信度  [obj_nums]
            - box_classes: 网络预测的box类别    [obj_nums]
    '''
    if onnx_path:
        # print(onnxruntime.get_device()) # GPU
        '''载入 onnx 模型，获取 ONNX Runtime 推理器'''
        providers = ['CPUExecutionProvider']
        if torch.cuda.is_available():
            providers.insert(0, 'CUDAExecutionProvider')
        ort_session = onnxruntime.InferenceSession(onnx_path, providers=providers)

    image = Image.open(img_path).convert('RGB')
    # Image 转numpy
    image = np.array(image)
    '''推理一张图像'''
    # xyxy
    boxes, box_scores, box_classes = model.onnxInfer(
        onnx_model=ort_session, 
        image=np.array(image), 
        img_size=img_size, 
        tf=tf, 
        device=device, 
        T=T, 
        image2color=image2color, 
        agnostic=agnostic, 
        vis_heatmap=vis_heatmap, 
        save_vis_path=save_vis_path, 
        )
    #  检测出物体才继续    
    if len(boxes) == 0: 
        print(f'no objects in image: {img_path}.')
        return boxes, box_scores, box_classes

    '''画框'''
    if save_vis_path!=None:
        # PltDrawBox(image, boxes, box_classes, box_scores, save_vis_path, image2color, class_names)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = OpenCVDrawBox(image, boxes, box_classes, box_scores, save_vis_path, image2color, class_names, resize_size=[2000, 2000], show_text=show_text)
        cv2.imwrite(save_vis_path, image)
        # 统计检测出的类别和数量
        detect_cls = dict(Counter(box_classes))
        detect_name = {}
        for key, val in detect_cls.items():
            detect_name[class_names[key]] = val
        print(f'detect result: {detect_name}')
    return boxes, box_scores, box_classes





def onnxInferenceVideo(model, device, class_names, image2color, img_size, tf, video_path, onnx_path, save_vis_path=None, T=0.3, agnostic=False, show_text=True):
    '''推理一段视频
        # Args:
            - device:        cpu/cuda
            - class_names:   每个类别的名称, list
            - image2color:   每个类别一个颜色
            - img_size:      固定图像大小 如[832, 832]
            - tf:            数据预处理(基于albumentation库)
            - video_pat:     视频路径
            - save_vis_path: 可视化图像保存路径
            - ckpt_path:     模型权重路径
            - T:             可视化的IoU阈值

        # Returns:
            - boxes:       网络回归的box坐标    [obj_nums, 4]
            - box_scores:  网络预测的box置信度  [obj_nums]
            - box_classes: 网络预测的box类别    [obj_nums]
    '''
    if onnx_path:
        # print(onnxruntime.get_device()) # GPU
        '''载入 onnx 模型，获取 ONNX Runtime 推理器'''
        providers = ['CPUExecutionProvider']
        if torch.cuda.is_available():
            providers.insert(0, 'CUDAExecutionProvider')
        ort_session = onnxruntime.InferenceSession(onnx_path, providers=providers)

    # 创建视频捕获对象
    cap = cv2.VideoCapture(video_path)
    # 获取视频的基本信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 定义视频编码器和创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 或者 'XVID' 'mp4v'
    out = cv2.VideoWriter(save_vis_path, fourcc, fps, (width, height))
    # 检查视频是否正确打开
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # 逐帧读取视频
    cnt_frame =  1
    start_time = time.time()
    while True:
        ret, frame = cap.read()  # ret是一个布尔值，frame是每一帧的图像
        if not ret:
            print("Reached end of video or failed to read frame.")
            break

        '''此处为推理'''
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        infer_s = time.time()
        boxes, box_scores, box_classes = model.onnxInfer(
            onnx_model=ort_session, 
            image=np.array(frame), 
            img_size=img_size, 
            tf=tf, 
            device=device, 
            T=T, 
            agnostic=agnostic, 
            )
        infer_e = time.time()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        #  检测出物体才继续    
        if len(boxes) != 0: 
            '''画框'''
            frame = OpenCVDrawBox(frame, boxes, box_classes, box_scores, None, image2color, class_names, resize_size=[2000, 2000], show_text=show_text)
            # 统计检测出的类别和数量
            detect_cls = dict(Counter(box_classes))
            detect_name = {}
            for key, val in detect_cls.items():
                detect_name[class_names[key]] = val
        # 写入处理后的帧到新视频
        out.write(frame)
        print(f'process frame {frame.shape}: [{cnt_frame}/{total_frames}] | time(ms): {round(infer_e - infer_s, 3)} | {detect_name}')
        cnt_frame += 1
    end_time = time.time()
    print(f"total_time: {end_time - start_time}(s) | fps: {cnt_frame / (end_time - start_time)}")
    # 释放视频捕获对象和视频写入对象，销毁所有OpenCV窗口
    cap.release()
    out.release()
    # cv2.destroyAllWindows()