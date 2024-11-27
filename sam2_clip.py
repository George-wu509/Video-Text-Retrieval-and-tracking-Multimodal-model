import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import random
import cv2
import clip
import numpy as np
import sys
import os

python_dir = "D:\Python"  # "C:\other\Python" or "D:\Python"

sys.path.append(os.path.abspath(python_dir + "/sam2"))
from sam2.build_sam import build_sam2_video_predictor

class VideoTextRetrievalTracker(nn.Module):
    def __init__(self, model_cfg, sam_checkpoint, device="cuda"):
        super(VideoTextRetrievalTracker, self).__init__()  # 首先調用 nn.Module 的初始化
        
        self.device = device
        self.predictor = build_sam2_video_predictor(model_cfg, sam_checkpoint, device=device)
        
        # Initialize CLIP
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        
        # Image preprocessing for CLIP
        self.clip_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                 (0.26862954, 0.26130258, 0.27577711))
        ])

    def init_state(self, video_path):
            return self.predictor.init_state(video_path)

    def add_new_points_or_box(self, inference_state, *args, **kwargs):
        return self.predictor.add_new_points_or_box(inference_state, *args, **kwargs)

    def propagate_in_video(self, inference_state):
        return self.predictor.propagate_in_video(inference_state)

    def encode_text(self, text):
        return self.clip_model.encode_text(clip.tokenize(text).to(self.device))

    def encode_image(self, image):
        """
        将输入图像编码为 CLIP 图像特征。

        Args:
        - image: 输入图像，可以是 numpy.ndarray 或 PIL.Image。

        Returns:
        - 图像特征向量。
        """
        # 确保输入是 numpy.ndarray 或 PIL.Image
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()  # 转为 numpy.ndarray
            if image.shape[0] == 3:  # 如果是 (3, H, W)，转换为 (H, W, 3)
                image = np.transpose(image, (1, 2, 0))
            image = Image.fromarray((image * 255).astype('uint8'))  # 转为 PIL.Image

        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'), 'RGB')  # 转为 PIL.Image

        # 应用 CLIP 的预处理
        with torch.no_grad():  # 禁用梯度计算
            image = self.clip_transform(image).unsqueeze(0).to(self.device)
            return self.clip_model.encode_image(image)

    def retrieve_frames(self, inference_state, text_query, batch_size=16):
        """
        检索与文本查询最相似的帧。

        Args:
        - inference_state: 包含视频帧的推理状态。
        - text_query: 查询文本。
        - batch_size: 每次处理的帧数量。

        Returns:
        - 帧的相似度向量。
        """
        text_features = self.encode_text(text_query)

        frame_features = []
        frames = inference_state["images"]

        # 分批处理帧
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            batch_features = []
            for frame in batch_frames:
                if frame.shape[2] == 4:  # 如果是 RGBA，转换为 RGB
                    frame = frame[:, :, :3]
                batch_features.append(self.encode_image(frame))

            # 拼接批次特征
            batch_features = torch.cat(batch_features, dim=0)
            frame_features.append(batch_features)

        # 合并所有批次
        frame_features = torch.cat(frame_features, dim=0)

        # 计算相似度
        with torch.no_grad():  # 禁用梯度计算
            similarity = (100.0 * frame_features @ text_features.T).softmax(dim=0)

        return similarity

    def multi_object_tracking(self, inference_state, text_queries, threshold, output_path):
        """
        追蹤多種類多物體，逐幀顯示分割遮罩和累積軌跡。

        Args:
        - inference_state: 推理狀態。
        - text_queries: 文本查詢列表（如 ["a red car", "a person", "a dog"]）。
        - threshold: 相似度閾值。
        - output_path: 輸出影片的目錄。
        """
        output_file = os.path.join(output_path, "prompt_tracking_video.mp4")

        similarities = [self.retrieve_frames(inference_state, query) for query in text_queries]

        # 建立顏色字典
        category_colors = {
            query: [random.randint(128, 255) for _ in range(3)] for query in text_queries
        }
        trajectory_colors = {
            query: [max(c // 2, 0) for c in color] for query, color in category_colors.items()
        }

        # 追蹤結果和軌跡
        tracked_objects = {query: [] for query in text_queries}
        object_tracks = {query: [] for query in text_queries}

        # 影片屬性
        frame_height, frame_width, _ = inference_state["images"][0].shape
        frame_rate = 30
        os.makedirs(output_path, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_video = cv2.VideoWriter(output_file, fourcc, frame_rate, (frame_width, frame_height))

        # 逐幀處理
        for frame_idx, frame in enumerate(inference_state["images"]):
            frame = process_frame(frame)  # 确保帧是 numpy.ndarray 且格式正确
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            for query_idx, (query, similarity) in enumerate(zip(text_queries, similarities)):
                if similarity[frame_idx] > threshold:
                    h, w = frame.shape[:2]
                    points = torch.tensor([[w / 2, h / 2]], device=self.device)
                    labels = torch.tensor([1], device=self.device)

                    with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
                        _, object_ids, masks = self.add_new_points_or_box(inference_state, frame_idx, 1, points, labels)

                    for obj_id, mask in zip(object_ids, masks):
                        mask_np = (mask.cpu().numpy() > 0).astype(np.uint8) * 255
                        mask_3d = cv2.merge([mask_np] * 3)

                        # 遮罩顏色
                        mask_color = np.zeros_like(frame_bgr, dtype=np.uint8)
                        mask_color[mask_np > 0] = category_colors[query]
                        frame_bgr = cv2.addWeighted(frame_bgr, 0.8, mask_color, 0.2, 0)

                        # 計算中心點
                        y, x = np.where(mask_np > 0)
                        if len(x) > 0 and len(y) > 0:
                            center_x = int(np.mean(x))
                            center_y = int(np.mean(y))
                            object_tracks[query].append((center_x, center_y))

            # 畫累積軌跡線
            for query, track in object_tracks.items():
                color = trajectory_colors[query]
                for i in range(1, len(track)):
                    cv2.line(frame_bgr, track[i - 1], track[i], color, thickness=1)

            out_video.write(frame_bgr)

        out_video.release()
        print(f"Video with segmentation and tracking saved to {output_file}")

def run_point_seg(predictor, inference_state, video_path, output_path, categories, frame_rate=30, vis_frame_stride=30):
    """
    追蹤多種類多物體的分割與軌跡顯示，逐幀累積軌跡。

    Args:
    - predictor: SAM2 預測器。
    - inference_state: 初始化的推理狀態。
    - video_path: 包含影格的資料夾路徑。
    - output_path: 保存輸出影片的目錄。
    - categories: 物體種類列表（如 ["person", "dog"]）。
    - frame_rate: 輸出影片的幀率。
    """
    output_file = os.path.join(output_path, "multi_point_tracking_video.mp4")

    # 建立每個種類的顏色字典
    category_colors = {
        category: [random.randint(0, 255) for _ in range(3)] for category in categories
    }
    trajectory_colors = {
        category: [max(c // 2, 0) for c in color] for category, color in category_colors.items()
    }

    # 初始化分割
    ann_frame_idx = 0
    objects = [
        {"id": 1, "category": "person", "points": np.array([[210, 350], [250, 220]], dtype=np.float32)},
        {"id": 2, "category": "dog", "points": np.array([[310, 250], [350, 220]], dtype=np.float32)},
    ]

    # 每物體初始化分割
    for obj in objects:
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=obj["id"],
            points=obj["points"],
            labels=np.array([1] * len(obj["points"]), dtype=np.int32),
        )

    # 分割傳播
    video_segments = {}
    object_tracks = {obj["id"]: [] for obj in objects}  # 每個物體的軌跡
    object_categories = {obj["id"]: obj["category"] for obj in objects}  # 每個物體的種類

    frame_files = sorted(
        [f for f in os.listdir(video_path) if f.endswith((".jpg", ".jpeg"))],
        key=lambda x: int(os.path.splitext(x)[0])
    )
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {}
        for i, out_obj_id in enumerate(out_obj_ids):
            mask = (out_mask_logits[i] > 0.0).cpu().numpy()

            # 確保 mask 是 2D
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = np.squeeze(mask, axis=0)

            video_segments[out_frame_idx][out_obj_id] = mask

            # 計算物體中心
            y, x = np.where(mask > 0)
            if len(x) > 0 and len(y) > 0:
                center_x = int(np.mean(x))
                center_y = int(np.mean(y))
                object_tracks[out_obj_id].append((center_x, center_y))  # 記錄中心位置

    # 構建影片輸出
    first_frame = cv2.imread(os.path.join(video_path, frame_files[0]))
    if first_frame is None:
        print(f"Error: Cannot read first frame in {video_path}")
        return
    frame_height, frame_width, _ = first_frame.shape

    os.makedirs(output_path, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(output_file, fourcc, frame_rate, (frame_width, frame_height))

    # 疊加遮罩和軌跡線並寫入影片
    for frame_idx, frame_file in enumerate(frame_files):
        frame_path = os.path.join(video_path, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: Cannot read frame {frame_file}")
            continue

        if frame_idx in video_segments:
            for obj_id, mask in video_segments[frame_idx].items():
                mask_3d = np.stack([mask] * 3, axis=-1)  # 將 mask 擴展為 (540, 960, 3)

                # 根據種類設置分割遮罩顏色
                category = object_categories[obj_id]
                mask_color = np.zeros_like(frame, dtype=np.uint8)
                mask_color[mask_3d[:, :, 0] > 0] = category_colors[category]
                frame = cv2.addWeighted(frame, 0.8, mask_color, 0.2, 0)

        # 添加累積軌跡線（逐幀增加）
        for obj_id, track in object_tracks.items():
            category = object_categories[obj_id]
            trajectory_color = trajectory_colors[category]
            for i in range(1, min(len(track), frame_idx + 1)):  # 畫到當前幀
                cv2.line(frame, track[i - 1], track[i], trajectory_color, thickness=1)

        out_video.write(frame)

    # 釋放資源
    out_video.release()
    print(f"Video with segmentation and tracking saved to {output_file}")

def run_box_seg(predictor, inference_state, video_path, output_path, categories, frame_rate=30, vis_frame_stride=30):
    """
    追蹤多種類多物體的分割與軌跡顯示，逐幀累積軌跡。

    Args:
    - predictor: SAM2 預測器。
    - inference_state: 初始化的推理狀態。
    - video_path: 包含影格的資料夾路徑。
    - output_path: 保存輸出影片的目錄。
    - categories: 物體種類列表（如 ["person", "dog"]）。
    - frame_rate: 輸出影片的幀率。
    """
    output_file = os.path.join(output_path, "multi_obj_tracking_video.mp4")

    # 建立每個種類的顏色字典
    category_colors = {
        category: [random.randint(0, 255) for _ in range(3)] for category in categories
    }
    trajectory_colors = {
        category: [max(c // 2, 0) for c in color] for category, color in category_colors.items()
    }

    # 初始化分割
    ann_frame_idx = 0
    objects = [
        {"id": 1, "category": "person", "box": np.array([300, 50, 500, 400], dtype=np.float32)},
        {"id": 2, "category": "person", "box": np.array([100, 100, 200, 300], dtype=np.float32)},
        {"id": 3, "category": "dog", "box": np.array([400, 300, 550, 450], dtype=np.float32)},
    ]

    # 每物體初始化分割
    for obj in objects:
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=obj["id"],
            box=obj["box"]
        )

    # 分割傳播
    video_segments = {}
    object_tracks = {obj["id"]: [] for obj in objects}  # 每個物體的軌跡
    object_categories = {obj["id"]: obj["category"] for obj in objects}  # 每個物體的種類

    frame_files = sorted(
        [f for f in os.listdir(video_path) if f.endswith((".jpg", ".jpeg"))],
        key=lambda x: int(os.path.splitext(x)[0])
    )
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {}
        for i, out_obj_id in enumerate(out_obj_ids):
            mask = (out_mask_logits[i] > 0.0).cpu().numpy()

            # 確保 mask 是 2D
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = np.squeeze(mask, axis=0)

            video_segments[out_frame_idx][out_obj_id] = mask

            # 計算物體中心
            y, x = np.where(mask > 0)
            if len(x) > 0 and len(y) > 0:
                center_x = int(np.mean(x))
                center_y = int(np.mean(y))
                object_tracks[out_obj_id].append((center_x, center_y))  # 記錄中心位置

    # 構建影片輸出
    first_frame = cv2.imread(os.path.join(video_path, frame_files[0]))
    if first_frame is None:
        print(f"Error: Cannot read first frame in {video_path}")
        return
    frame_height, frame_width, _ = first_frame.shape

    os.makedirs(output_path, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(output_file, fourcc, frame_rate, (frame_width, frame_height))

    # 疊加遮罩和軌跡線並寫入影片
    for frame_idx, frame_file in enumerate(frame_files):
        frame_path = os.path.join(video_path, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: Cannot read frame {frame_file}")
            continue

        if frame_idx in video_segments:
            for obj_id, mask in video_segments[frame_idx].items():
                mask_3d = np.stack([mask] * 3, axis=-1)  # 將 mask 擴展為 (540, 960, 3)

                # 根據種類設置分割遮罩顏色
                category = object_categories[obj_id]
                mask_color = np.zeros_like(frame, dtype=np.uint8)
                mask_color[mask_3d[:, :, 0] > 0] = category_colors[category]
                frame = cv2.addWeighted(frame, 0.8, mask_color, 0.2, 0)

        # 添加累積軌跡線（逐幀增加）
        for obj_id, track in object_tracks.items():
            category = object_categories[obj_id]
            trajectory_color = trajectory_colors[category]
            for i in range(1, min(len(track), frame_idx + 1)):  # 畫到當前幀
                cv2.line(frame, track[i - 1], track[i], trajectory_color, thickness=1)

        out_video.write(frame)

    # 釋放資源
    out_video.release()
    print(f"Video with segmentation and tracking saved to {output_file}")

def run_prompt_seg(predictor, inference_state, video_path, output_path, text_queries, text_threshold):
   
    tracked_objects = predictor.multi_object_tracking(inference_state, text_queries, text_threshold, output_path)
    #visualize_results(inference_state, tracked_objects)

def show_video(video_dir, frame_rate=30):
    """
    在 VSCode 中播放影格序列作為影片。

    Args:
    - video_dir (str): 包含影格的目錄。
    - frame_rate (int): 播放每秒影格數（默認 30fps）。
    """
    # 掃描並排序所有影格檔案
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # 播放影格
    plt.figure(figsize=(9, 6))
    for frame_idx, frame_name in enumerate(frame_names):
        frame_path = os.path.join(video_dir, frame_name)
        frame = Image.open(frame_path)

        plt.clf()  # 清除上一影格
        plt.imshow(frame)
        plt.axis('off')
        plt.title(f"Frame {frame_idx + 1}")
        plt.pause(1 / frame_rate)  # 以指定幀率暫停
    plt.show()

def save_video(video_dir, output_path, frame_rate=30, resize=None):
    """
    將影格序列轉換為 MP4 文件。

    Args:
    - video_dir (str): 包含影格的目錄。
    - output_path (str): 輸出的 MP4 文件路徑（包括文件名）。
    - frame_rate (int): 影片幀率（默認 30fps）。
    - resize (tuple): 影格的目標大小 (width, height)，默認不縮放。
    """
    output_file = output_path + "output_video.mp4"

    # 掃描並排序所有影格檔案
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    if not frame_names:
        print("No frames found in the directory.")
        return

    # 讀取第一張影格以獲取影片尺寸
    first_frame_path = os.path.join(video_dir, frame_names[0])
    first_frame = cv2.imread(first_frame_path)
    if first_frame is None:
        print(f"Failed to read the first frame: {first_frame_path}")
        return

    # 如果需要，縮放影格大小
    if resize:
        width, height = resize
    else:
        height, width, _ = first_frame.shape

    # 初始化 VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 指定 MP4 格式
    video_writer = cv2.VideoWriter(output_file, fourcc, frame_rate, (width, height))

    # 遍歷影格並寫入影片
    for frame_name in frame_names:
        frame_path = os.path.join(video_dir, frame_name)
        frame = cv2.imread(frame_path)

        if frame is None:
            print(f"Failed to read frame: {frame_path}")
            continue

        # 如果需要，縮放影格大小
        if resize:
            frame = cv2.resize(frame, (width, height))

        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved to {output_file}")

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def process_frame(frame):
    """
    确保输入帧是 numpy.ndarray 格式并具有正确的维度。
    """
    # 如果是 torch.Tensor，转换为 numpy.ndarray
    if isinstance(frame, torch.Tensor):
        frame = frame.cpu().numpy()
        if frame.shape[0] == 3:  # 如果是 (3, H, W)，转换为 (H, W, 3)
            frame = np.transpose(frame, (1, 2, 0))

    # 如果是 PIL.Image，转换为 numpy.ndarray
    elif isinstance(frame, Image.Image):
        frame = np.array(frame)

    # 确保是 RGB 格式
    if len(frame.shape) == 2:  # 如果是灰度图像
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif frame.shape[2] == 1:  # 如果是单通道
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

    return frame

def visualize_results(inference_state, tracked_objects):
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]
    
    for frame_idx, frame in enumerate(inference_state["images"]):
        img = frame.copy()
        for query_idx, (query, objects) in enumerate(tracked_objects.items()):
            color = colors[query_idx % len(colors)]
            for _, obj_id, mask in objects:
                if _ == frame_idx:
                    mask_visualized = (mask.astype(np.uint8) * 255)[:,:,None] * np.array(color)
                    img = cv2.addWeighted(img, 1, mask_visualized.astype(np.uint8), 0.5, 0)
            
            cv2.putText(img, query, (10, 30 + query_idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.imshow(f'Frame {frame_idx + 1}', img[:,:,::-1])
        cv2.waitKey(1)
    
    cv2.destroyAllWindows()

def check_device():
    if torch.cuda.is_available:
        device = "cuda"
    else:
        device = "cpu"
    return device

# Usage example
if __name__ == "__main__":

    # Model setting
    sam_checkpoint = "./model/sam2.1_hiera_tiny.pt" #"./model/sam2.1_hiera_large.pt"
    model_cfg = python_dir + "/sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml" # "/sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml"
    video_path = "./image/bedroom"  # "./image/sav_049989.mp4"
    output_path = "./output/"
    video_max_frame=5 #None

    # Job setting
    job_index = 3  #1: seg using point, 2:seg using box, 3: seg using prompt
    text_queries = ["a red car", "a person", "a dog"]
    categories=["person", "dog"]
    text_threshold = 0.5

    # -------------------------------------------------------

    # Create VideoTextRetrievalTracker
    sysdevice = check_device()
    predictor = VideoTextRetrievalTracker(model_cfg, sam_checkpoint, device = sysdevice)

    # Initialize the state using SAM2's init_state function
    #with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    inference_state = predictor.init_state(video_path)
    
    if job_index == 1:
        run_point_seg(predictor, inference_state, video_path, output_path, categories)
    elif job_index == 2:
        run_box_seg(predictor, inference_state, video_path, output_path, categories)
    elif job_index == 3:
        run_prompt_seg(predictor, inference_state, video_path, output_path, text_queries, text_threshold)
    elif job_index == 4:
        show_video(video_path, frame_rate=30)
    elif job_index == 5:
        save_video(video_path, output_path, frame_rate=24, resize=(1920, 1080))

