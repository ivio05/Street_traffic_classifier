import cv2
import torch
from torchvision.transforms import v2
import torch.nn.functional as nnf

from components import inter_p, find_boxes
from model import Net

IMAGE_SIZE = (224,224)
THRESH = 45
NET_PER_FRAMES = 3
THRESH_PROB = 0.5
PP = 10
MIN_SIZE = 20
START_FRAME = 0
FINISH_FRAME = 0

MODEL_NAME = 'best_model.pt'
VIDEO_NAME = 'cctvs/cctv1.mp4'
BGR_NAME = 'backgrounds/bgr1.jpg'
RESULT_NAME = 'results/cctv1_result.mp4'
device = 'cpu'

class_color = {0 : (0, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255)}
class_name = {0 : 'noise', 1: 'person', 2: 'car'}

transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize(IMAGE_SIZE),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

net = Net()
net.load_state_dict(torch.load(MODEL_NAME, weights_only=True, map_location=torch.device(device)))
net.eval()

bgr = cv2.imread(BGR_NAME)
grayBgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

cap = cv2.VideoCapture(VIDEO_NAME)
cnt = 0

if not cap.isOpened():
    print("Cannot open video")
    exit()

beforeRects = []
before_classes = []
before_probs = []

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourCC = cv2.VideoWriter_fourcc(*'mp4v')
result = cv2.VideoWriter(RESULT_NAME, fourCC, 20, (w, h))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cnt += 1
    
    print(cnt)
    
    if cnt < START_FRAME and cnt != FINISH_FRAME:
        continue
    
    if (cnt - 1) % NET_PER_FRAMES == 0:
        beforeRects = []
        before_classes = []
        before_probs = []
        
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(grayFrame, grayBgr)
        ret, mask = cv2.threshold(diff, THRESH, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boundRect = [None]*len(contours)
        boxes = find_boxes(frame, contours, PP, MIN_SIZE)
        
        for left, right, up, down, img, boundRect in boxes:
            output = net(transforms(torch.tensor([img]).permute(0, 3, 1, 2)))
            prob = nnf.softmax(output, dim=1)
            top_p, top_class = prob.topk(1, dim = 1)
            if top_p > THRESH_PROB and top_class != 0:
                
                flag = 0
                for box in boxes:
                    if inter_p([left, right, up, down], box) > 0.7:
                        flag = 1
                        break

                if flag:
                    continue

                beforeRects.append(boundRect)
                before_classes.append(top_class)
                before_probs.append(top_p)
    
    for i in range(len(beforeRects)):
        boundRect = beforeRects[i]
        top_class = before_classes[i]
        top_p = before_probs[i]
            
        color = class_color[top_class.item()]
        cv2.rectangle(frame, (int(boundRect[0]), int(boundRect[1])), (int(boundRect[0]+boundRect[2]), int(boundRect[1]+boundRect[3])), color, 2)
        cv2.putText(frame, f'{class_name[top_class.item()]} : {str(top_p.item())[:5]}', (int(boundRect[0]+boundRect[2]), int(boundRect[1])), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=color)
    
    result.write(frame)
    # cv2.imshow('bg', frame)
    # cv2.waitKey(30)

cv2.destroyAllWindows()
cap.release()
result.release()