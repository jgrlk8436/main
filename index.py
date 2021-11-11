#引入的library
import streamlit as st
from typing import List
import cv2
import torch
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.colors as mcolors
from PIL import Image
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer,ClientSettings

# dlib部分
import face_recognition as fr
import os
import pickle
import datetime
import csv
import base64
import base64
from io import BytesIO





CLASSES = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 
            'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 
            'scissors', 'teddy bear', 'hair drier', 'toothbrush' ]
CLASSES2 = [ 'dog','person','cat','tv','car','meatballs','marinara sauce','tomato soup','chicken noodle soup','french onion soup','chicken breast'
,'ribs','pulled pork','hamburger','cavity','awake','drowsy']  


WEBRTC_CLIENT_SETTINGS = ClientSettings(
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )


st.set_page_config(
    page_title="Image processing",
)




@st.cache(max_entries=2)
def get_yolo5(model_type='s'):

    return torch.hub.load('ultralytics/yolov5', 
                          'yolov5{}'.format(model_type), 
                          pretrained=True
                          )


# yolov5 drowsy
@st.cache(max_entries=5)
def get_yolo555():

    return torch.hub.load('ultralytics/yolov5', 
                          'custom',
                          path='yolov5/runs/train/exp4/weights/last.pt',
                          force_reload=True
                          )




@st.cache(max_entries=10)
def get_preds(img : np.ndarray) -> np.ndarray:

    return model([img]).xyxy[0].cpu().numpy()











def get_colors(indexes : List[int]) -> dict:
   
    to_255 = lambda c: int(c*255)
    tab_colors = list(mcolors.TABLEAU_COLORS.values())
    tab_colors = [list(map(to_255, mcolors.to_rgb(name_color))) 
                                                for name_color in tab_colors]
    base_colors = list(mcolors.BASE_COLORS.values())
    base_colors = [list(map(to_255, name_color)) for name_color in base_colors]
    rgb_colors = tab_colors + base_colors
    rgb_colors = rgb_colors*5

    color_dict = {}
    for i, index in enumerate(indexes):
        if i < len(rgb_colors):
            color_dict[index] = rgb_colors[i]
        else:
            color_dict[index] = (255,0,0)

    return color_dict


def get_colors2(indexes : List[int]) -> dict:
   
    to_255 = lambda c: int(c*255)
    tab_colors2 = list(mcolors.TABLEAU_COLORS.values())
    tab_colors2 = [list(map(to_255, mcolors.to_rgb(name_color))) 
                                                for name_color in tab_colors2]
    base_colors2 = list(mcolors.BASE_COLORS.values())
    base_colors2 = [list(map(to_255, name_color)) for name_color in base_colors2]
    rgb_colors2 = tab_colors2 + base_colors2
    rgb_colors2 = rgb_colors2*5

    color_dict2 = {}
    for i, index in enumerate(indexes):
        if i < len(rgb_colors2):
            color_dict2[index] = rgb_colors2[i]
        else:
            color_dict2[index] = (255,0,0)

    return color_dict2




def get_legend_color(class_name : int):
   

    index = CLASSES.index(class_name)
    color = rgb_colors[index]
    return 'background-color: rgb({color[0]},{color[1]},{color[2]})'.format(color=color)


# 新增class 2


def get_legend_color2(class_name2 : int):
   

    index2 = CLASSES2.index(class_name2)
    color2 = rgb_colors2[index2]
    return 'background-color: rgb({color[0]},{color[1]},{color[2]})'.format(color=color2)








class VideoTransformer(VideoTransformerBase):

   
    def __init__(self):
        self.model = model
        self.rgb_colors = rgb_colors
        self.target_class_ids = target_class_ids

    def get_preds(self, img : np.ndarray) -> np.ndarray:
        return self.model([img]).xyxy[0].cpu().numpy()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = self.get_preds(img)
        result = result[np.isin(result[:,-1], self.target_class_ids)]
        
        for bbox_data in result:
            xmin, ymin, xmax, ymax, _, label = bbox_data
            p0, p1, label = (int(xmin), int(ymin)), (int(xmax), int(ymax)), int(label)
            img = cv2.rectangle(img, 
                                    p0, p1, 
                                    self.rgb_colors[label], 2) 

        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)




#define dlib face

def authenticate():

 def classify_face(img):
    faces = pickle.loads(open('9save','rb').read())

    face_locations = fr.face_locations(img)
    unknown_face_encodings = fr.face_encodings(img,face_locations)
    face_names = []
    date_time_list=[]

    for face_encodings in unknown_face_encodings:
        matches = fr.compare_faces(faces["faces_encoded"],face_encodings)
        name = "Unknown"
        face_distances = fr.face_distance(faces["faces_encoded"],face_encodings)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = faces["known_face_names"][best_match_index]
        face_names.append(name)

        now=datetime.datetime.now()
        dtString=now.strftime('%A,%d %B %Y (IST)  %H:%M:%S')
        date_time_list.append(dtString)

        
        
        for (top,right,bottom,left),name in zip(face_locations,face_names):
            cv2.rectangle(img, (left-20, top -20),(right+20,bottom+20), (255,0,0),2)
            cv2.rectangle(img,(left-20,bottom-15),(right+20,bottom+20),(255,0,0),cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img,name,(left-20,bottom+15),font,1.0, (255,255,255),2)
    # return img
    return img,face_names,date_time_list

 def csvdata(x,y):
    rows=[[x,y]]
    f=open('Employee_Details.csv','a')
    with f:
        csvwriter=csv.writer(f)
        csvwriter.writerows(rows)
   
        
 cap = cv2.VideoCapture(0)
#1 for External Webcam
 face_names2 =[]
 date_time_list2=[]
 while True: 
        
        ret,img = cap.read()
        img,face_names,date_time_list=classify_face(img)
        cv2.imshow('Face_Recognition',img)
        if face_names!=[]:
          for i in face_names:
             if i=="Unknown":
               print(i)
               print(date_time_list)
               st.text(i)
               st.text(date_time_list)
               csvdata(i,date_time_list)
             else:
               print("{} You are Login".format(i))
               print(date_time_list)
               st.text(i+" You are Login")
               face_names2.append(i)
               date_time_list2.append(date_time_list)
               st.text(date_time_list)
               csvdata(i,date_time_list)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        def get_table_download_link_csv(df): 
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(
            csv.encode()).decode() 
            return f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv">Download csv file</a>'
        
        
 
 cap.release()    
 cv2.destroyAllWindows()
 test = pd.DataFrame({'names':face_names2,'date_time':date_time_list2})
 st.markdown(get_table_download_link_csv(test), unsafe_allow_html=True)




model_type = st.sidebar.selectbox(
    'Select model type',
    ('s', 'm', 'l', 'x'),
    index=1,
    format_func=lambda s: s.upper())


# 新增 yolov5 drowsy
with st.spinner('Loading the model...'):
    model = get_yolo5(model_type)
    model2 = get_yolo555()
st.success('Loading the model.. Done!')

prediction_mode = st.sidebar.radio(
    "",
    ('Single image', 'Web camera','dlib face recognition(real time)','dlib face recognition(image)','yolov5(awake drowsy)'),
    index=0)
    
classes_selector = st.sidebar.multiselect('yolov5 (web camera) Select classes', 
                                        CLASSES, default='person')
all_labels_chbox = st.sidebar.checkbox('All classes', value=False)




# 新增class2
classes_selector2 = st.sidebar.multiselect('yolov5 (awake and drowsy) Select classes', 
                                        CLASSES2, default='person')
all_labels_chbox2 = st.sidebar.checkbox('All classes2', value=False)



# # Exam
# text_type = st.sidebar.selectbox(
#     'Select Exam ',
#     ('Exam A', 'Exam B', 'Exam C', 'Exam D'),
#     index=1,
#     format_func=lambda s: s.upper())





if all_labels_chbox:
    target_class_ids = list(range(len(CLASSES)))
elif classes_selector:
    target_class_ids = [CLASSES.index(class_name) for class_name in classes_selector]
else:
    target_class_ids = [0]



#新增class2
if all_labels_chbox2:
    target_class_ids2 = list(range(len(CLASSES2)))
elif classes_selector2:
    target_class_ids2 = [CLASSES2.index(class_name) for class_name in classes_selector2]
else:
    target_class_ids2 = [0]




rgb_colors = get_colors(target_class_ids)
detected_ids = None

rgb_colors2 = get_colors2(target_class_ids2)
detected_ids2 = None

if prediction_mode == 'Single image':
    st.title('YOLOv5 demo (picture)')
   
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=['png', 'jpg', 'jpeg'])

  
    if uploaded_file is not None:

    
        bytes_data = uploaded_file.getvalue()
        file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = get_preds(img)

        
        result_copy = result.copy()
       
        result_copy = result_copy[np.isin(result_copy[:,-1], target_class_ids)]
        

        detected_ids = []
        
        img_draw = img.copy().astype(np.uint8)
        
        for bbox_data in result_copy:
            xmin, ymin, xmax, ymax, _, label = bbox_data
            p0, p1, label = (int(xmin), int(ymin)), (int(xmax), int(ymax)), int(label)
            img_draw = cv2.rectangle(img_draw, 
                                    p0, p1, 
                                    rgb_colors[label], 2) 
            detected_ids.append(label)
        
      
        st.image(img_draw, use_column_width=True)

    detected_ids = set(detected_ids if detected_ids is not None else target_class_ids)
    labels = [CLASSES[index] for index in detected_ids]
    legend_df = pd.DataFrame({'label': labels})
    st.dataframe(legend_df.style.applymap(get_legend_color))


elif prediction_mode == 'Web camera':
    st.title('YOLOv5 demo (real time)')
 
    ctx = webrtc_streamer(
        key="example", 
        video_transformer_factory=VideoTransformer,
        client_settings=WEBRTC_CLIENT_SETTINGS,)

 
    if ctx.video_transformer:
        ctx.video_transformer.model = model
        ctx.video_transformer.rgb_colors = rgb_colors
        ctx.video_transformer.target_class_ids = target_class_ids

    detected_ids = set(detected_ids if detected_ids is not None else target_class_ids)
    labels = [CLASSES[index] for index in detected_ids]
    legend_df = pd.DataFrame({'label': labels})
    st.dataframe(legend_df.style.applymap(get_legend_color))

#dlib face

elif prediction_mode == 'dlib face recognition(real time)':
    
    def main():
        html_temp = """
        <div style="background-color:tomato;padding:10px">
        <h2 style="color:white;text-align:center;">Face recognition</h2>
        </div>
        """
        
        st.markdown(html_temp,unsafe_allow_html=True)
        st.text("\n")
        
        col1, col2, col3,col4,col5 = st.beta_columns(5)
        if col3.button("start"):
            st.text("Attendance results")
            authenticate()
    main()   
    
elif prediction_mode == 'dlib face recognition(image)':
    st.title('Face recognition (picture)')
    uploaded_fileA = st.file_uploader(
        "Choose an image",
        type=['png', 'jpg', 'jpeg'])

    if uploaded_fileA is not None:

    
        bytes_data = uploaded_fileA.getvalue()
        file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img,use_column_width=True)
        # result = get_preds(img)

        
        # result_copy = result.copy()
       
        # result_copy = result_copy[np.isin(result_copy[:,-1], target_class_ids)]
        

        # detected_ids = []
        
        # img_draw = img.copy().astype(np.uint8)
        # for bbox_data in result_copy:
        #     xmin, ymin, xmax, ymax, _, label = bbox_data
        #     p0, p1, label = (int(xmin), int(ymin)), (int(xmax), int(ymax)), int(label)
        #     img_draw = cv2.rectangle(img_draw, 
        #                             p0, p1, 
        #                             rgb_colors[label], 2) 
        #     detected_ids.append(label)
        
      
        # st.image(img_draw, use_column_width=True)

# detected_ids = set(detected_ids if detected_ids is not None else target_class_ids)
# labels = [CLASSES[index] for index in detected_ids]
# legend_df = pd.DataFrame({'label': labels})
# st.dataframe(legend_df.style.applymap(get_legend_color))


elif prediction_mode == 'yolov5(awake drowsy)':
    st.title('YOLOv5 (awake drowsy)')
    ctx = webrtc_streamer(
        key="example", 
        video_transformer_factory=VideoTransformer,
        client_settings=WEBRTC_CLIENT_SETTINGS,)

    if ctx.video_transformer:
        ctx.video_transformer.model = model2
        ctx.video_transformer.rgb_colors = rgb_colors2
        ctx.video_transformer.target_class_ids = target_class_ids2


    detected_ids2 = set(detected_ids2 if detected_ids2 is not None else target_class_ids2)
    labels = [CLASSES2[index] for index in detected_ids2]
    legend_df = pd.DataFrame({'label': labels})
    st.dataframe(legend_df.style.applymap(get_legend_color2))



