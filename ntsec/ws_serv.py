# TouchDesigner Web Server DAT

import json
import cv2
import numpy as np
from urllib.parse import urlparse
import mimetypes
import os

ptsGLOBE = []  # 宣告全域變數


# 指定靜態檔案的根目錄，請根據實際情況修改
web_root =  os.path.join(project.folder, 'docs')
print(web_root)

# Global variable to store previous circle positions
previous_circles = []

# ===== 圓形偵測 =====
# ===== 圓形偵測（最終版，兼容舊版 TD） =====
def detect_circles(top_name='null1'):
    top = op(top_name)
    if not top or not top.valid:
        return []                                # 找不到 TOP

    # 取 numpy array，float32, 0~1
    arr = top.numpyArray(delayed=False)
    if arr is None or arr.size == 0:
        return []                                # TOP 還沒出畫面

    # 轉成 uint8 0~255，OpenCV 才吃得下
    img = (arr * 255).clip(0, 255).astype(np.uint8)   # shape (H, W, 4)

    h, w = img.shape[:2]

    # 灰階 + 去噪
    gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)     # (H, W) uint8
    gray = cv2.medianBlur(gray, 5)

    # Hough 圓形偵測
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(10, int(min(h, w) * 0.05)),
        param1=100,
        param2=30,
        minRadius=5,
        maxRadius=0)

    pts = []
    if circles is not None:
        for x, y, r in np.round(circles[0]).astype(int):
            pts.append([x / w, y / h])           # 0~1 正規化
    return pts

# Function to track circles and calculate movement
def track_circles(new_circles):
    global previous_circles
    result = []
    
    if not previous_circles:
        # First detection - no movement data yet
        for pos in new_circles:
            result.append({"pos": pos, "movement": [0, 0]})
    else:
        # Match new circles with previous ones based on proximity
        matched_indices = []
        
        for pos in new_circles:
            best_match = None
            min_distance = float('inf')
            matched_idx = -1
            
            # Find the closest previous circle
            for i, prev_pos in enumerate(previous_circles):
                if i in matched_indices:
                    continue  # Skip already matched points
                
                # Calculate Euclidean distance
                dist = np.sqrt((pos[0] - prev_pos[0])**2 + (pos[1] - prev_pos[1])**2)
                if dist < min_distance and dist < 0.1:  # Threshold for matching
                    min_distance = dist
                    best_match = prev_pos
                    matched_idx = i
            
            if best_match is not None:
                # Calculate movement vector
                movement = [pos[0] - best_match[0], pos[1] - best_match[1]]
                result.append({"pos": pos, "movement": movement})
                matched_indices.append(matched_idx)
            else:
                # New circle with no match
                result.append({"pos": pos, "movement": [0, 0]})
    
    # Update previous circles for next detection
    previous_circles = new_circles.copy()
    return result

# ===== HTTP (保持原本功能) =====
def onHTTPRequest(webServerDAT, request, response):
    parsed_url = urlparse(request['uri'])
    path = parsed_url.path

    # 如果請求的是根目錄，返回 index.html
    if path == '/':
        path = '/index.html'

    # 不檢查目錄遍歷，直接拼接路徑
    safe_path = os.path.join(web_root, path.lstrip('/'))

    try:
        # 嘗試以文字模式讀取檔案
        with open(safe_path, 'r', encoding='utf-8') as f:
            response['data'] = f.read()
            response['statusCode'] = 200
            response['statusReason'] = 'OK'
            mime_type, _ = mimetypes.guess_type(safe_path)
            if mime_type:
                response['Content-Type'] = mime_type
    except UnicodeDecodeError:
        try:
            # 若為二進位檔案（如圖片）
            with open(safe_path, 'rb') as f:
                response['data'] = f.read()
                response['statusCode'] = 200
                response['statusReason'] = 'OK'
                mime_type, _ = mimetypes.guess_type(safe_path)
                if mime_type:
                    response['Content-Type'] = mime_type
        except Exception as e:
            response['statusCode'] = 500
            response['statusReason'] = 'Internal Server Error'
            response['data'] = f'500 Internal Server Error: {e}'
    except FileNotFoundError:
        response['statusCode'] = 404
        response['statusReason'] = 'Not Found'
        response['data'] = '404 Not Found'
    except Exception as e:
        response['statusCode'] = 500
        response['statusReason'] = 'Internal Server Error'
        response['data'] = f'500 Internal Server Error: {e}'

    return response



# ===== WebSocket callbacks =====
def onWebSocketOpen(webServerDAT, client, uri):
    return

def onWebSocketClose(webServerDAT, client):
    return

def onWebSocketReceiveText(webServerDAT, client, data):
    global ptsGLOBE  # 明確宣告使用全域變數
    msg = data.strip()

    if msg.lower() == 'detect':
        #pts = detect_circles()                   # [[nx, ny], ...]
        tracked_circles = track_circles(ptsGLOBE)     # Track circles and calculate movement
        webServerDAT.webSocketSendText(client, json.dumps(tracked_circles))
    elif msg.lower().startswith('set'):
        try:
            # 嘗試從"set"之後的部分載入JSON資料
            payload = msg[3:].strip()
            pts = json.loads(payload)
            if isinstance(pts, list) and all(isinstance(p, list) and len(p) == 2 for p in pts):
                ptsGLOBE = pts
                webServerDAT.webSocketSendText(client, "ptsGLOBE updated")
            else:
                webServerDAT.webSocketSendText(client, "Invalid format: expecting List[List[float]]")
        except json.JSONDecodeError:
            webServerDAT.webSocketSendText(client, "Invalid JSON payload")
    else:
        # 其他訊息照原樣回傳
        webServerDAT.webSocketSendText(client, data)

    return

def onWebSocketReceiveBinary(webServerDAT, client, data):
    webServerDAT.webSocketSendBinary(client, data)
    return

def onWebSocketReceivePing(webServerDAT, client, data):
    webServerDAT.webSocketSendPong(client, data=data)
    return

def onWebSocketReceivePong(webServerDAT, client, data):
    return

def onServerStart(webServerDAT):
    return

def onServerStop(webServerDAT):
    return
