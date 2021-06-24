import cv2
from keras.models import load_model
import os
import numpy as np
import streamlit as st
from tensorflow.keras.applications.resnet50 import ResNet50
import io
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import tempfile
from PIL import Image
import pandas as pd
from googlesearch import search
import requests
import time
import os
from keras.layers.convolutional import Conv2D, MaxPooling2D, SeparableConv2D
from keras.layers.core import Dense, Flatten, Dropout
# 设置标题
def save_uploadedfile(uploadedfile):
     with open(os.path.join("tempDir",uploadedfile.name),"wb") as f:
         f.write(uploadedfile.getbuffer())




#st.title('人工智慧期末作品-物件分類')
rad = st.sidebar.radio("導覽頁",["Home","Demo","Model","Source code","結論","參考來源"])
if rad=="Home":
    st.markdown("""
    ### S10659032 資工系 黃靖傑
    # 一、簡介
    1.動機:在台南這有許久歷史的古都，有些時候在路上可能會看到一些美食或景點，卻可能因為不知道叫什麼名子，而無法如何購買或尋找相關歷史資訊。

    2.目的: 透過物件辨識分類的功能可以幫助使用者快速收尋到想要的資訊。(類似google鏡頭)
    # 技術探討與應⽤
    1.欲使⽤的 AI 技術介紹:  
    ans: 捲積神經網路(CNN) ,遷移學習(transfer-learning) 

    2.有哪些⽅法？技術難度/優劣勢?  
    ans: VGG16,VGG19,resNet  
    - 困難點:與前端API串接方法?(GCP、herkou、區網?)
    - 優勢:模型眾多、且部分github直接附上trained weight
    # 研究⽅法
    1.參考/修改哪些程式？  
    ans: keras函式庫中提供訓練過的Imagenet的模型  

    2.技術的特徵與難度  
    ans:串接一堆API很麻煩，而且載入過多library導致系統變慢  
    """)
elif rad=="Demo":
    app_mode = st.selectbox("切換模型:",
            ["自訓練台南美食", "keras ResNet50"])
        
    # 項目一
    if app_mode == "自訓練台南美食":
        st.markdown("""分類類別0:牛肉湯,1:水果冰,2:意麵,3:壽司,4:臭豆腐  """)
        Tainanmodel = load_model('623358fin5food_vgg16.h5')
        uploaded_file1 = st.file_uploader("上傳圖片", type="jpg")
        
        if uploaded_file1 is not None:
            # 将传入的文件转为Opencv格式
            file_detail = {"FileName":uploaded_file1.name}
            file_bytes = np.asarray(bytearray(uploaded_file1.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            # 展示图片
            st.image(opencv_image, channels="BGR")
            st.write(uploaded_file1.name)
 
            if st.button("分類"):
                with tempfile.TemporaryDirectory() as dirname:
                    with open (os.path.join(dirname, uploaded_file1.name),"wb") as f:
                        #f.write(uploaded_file1.getbuffer())
                    
                        #img = image.load_img(os.path.join(dirname,uploaded_file1.name), target_size=(224, 224))
                        image_pred = cv2.cvtColor(file_bytes,cv2.COLOR_BGR2RGB)
                        image_pred = np.array(image_pred) / 255.0
                        image_pred = cv2.resize(image_pred, (224, 224))
                        #x = image.img_to_array(img)
                        x = np.expand_dims(image_pred, axis=0)
                        x = preprocess_input(x)
                        #{'beaf': 0, 'ice': 1, 'noodle': 2, 'susi': 3, 'tofu': 4}
                        preds = Tainanmodel.predict(x)
                        
                        st.write('Predicted:',preds)
                        y = np.transpose(preds)
                        
                        #print(type(preds))
                        pred_index = np.argmax(preds,axis=1)
                        # st.text(pred_index)
                        if pred_index==0:
                            st.write("牛肉湯")
                        elif pred_index==1:
                            st.write("水果冰")
                        elif pred_index==2:
                            st.write("意麵")
                        elif pred_index==3:
                            st.write("壽司")
                        elif pred_index==4:
                            st.write("臭豆腐")
                        pred_text  = '準確度: '+ str(y[np.argmax(preds,axis=1)[0]])
                        st.text(pred_text)
    # 項目二
    # elif app_mode == "自訓練kaggle食物訓練集":
        
    #     # sum2 = model2.summary()
    #     # st.write(sum2)
    #     uploaded_file = st.file_uploader("上传一张图片", type="jpg")
        
    #     if uploaded_file is not None:
    #         # 将传入的文件转为Opencv格式
    #         file_detail = {"FileName":uploaded_file.name}
    #         file_bytes2 = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    #         opencv_image = cv2.imdecode(file_bytes2, 1)
    #         # 展示图片
    #         st.image(opencv_image, channels="BGR",clamp=True)
            
    #         if st.button("分類"):
    #             with tempfile.TemporaryDirectory() as dirname:

    #                 with open (os.path.join(dirname, uploaded_file.name),"wb") as f:
    #                     f.write(uploaded_file.getbuffer())

    #                     #img = image.load_img(os.path.join(dirname,uploaded_file.name), target_size=(224, 224))
    #                     #x = image.img_to_array(img)
    #                     st.write(uploaded_file.name)
    #                     image_pred = cv2.cvtColor(file_bytes2,cv2.COLOR_BGR2RGB)
    #                     image_pred = np.array(image_pred) / 255.0
    #                     x = cv2.resize(image_pred, (224, 224))
    #                     x = np.expand_dims(x, axis=0)
                        
    #                     x = preprocess_input(x)

    #                     # st.write(x.shape)
    #                     preds2 = model2.predict(x)
    #                     st.write(preds2)
    #                     st.write(np.argmax(preds2))

        

    elif app_mode == "keras ResNet50":
        model3 = ResNet50(weights='imagenet')
        # 上传图片并展示
        uploaded_file = st.file_uploader("上传一张图片", type="jpg")
        
        if uploaded_file is not None:
            # 将传入的文件转为Opencv格式
            file_detail = {"FileName":uploaded_file.name}
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            # 展示图片
            st.image(opencv_image, channels="BGR")
            st.write(uploaded_file.name)
            with tempfile.TemporaryDirectory() as dirname:
            # print('暫存目錄：', dirname)
            # tempDir = "D:/"
                with open (os.path.join(dirname, uploaded_file.name),"wb") as f:
                    f.write(uploaded_file.getbuffer())

                    #img = image.load_img(os.path.join(dirname,uploaded_file.name), target_size=(224, 224)) #不要刪
                    dataBytesIO = io.BytesIO(uploaded_file.getbuffer())
                    img = Image.open(dataBytesIO)
                    x = image.img_to_array(img)
                    x = np.expand_dims(x, axis=0)
                    x = preprocess_input(x)
                    
                    preds = model3.predict(x)
                    # decode the results into a list of tuples (class, description, probability)
                    # (one such list for each sample in the batch)
                    st.write('Predicted:', decode_predictions(preds, top=3)[0])
                        # 保存图片
        #cv2.imwrite('test.jpg',opencv_image)

    #####################################################
   
elif rad=="Model":
    st.markdown("""## fine tuned
    透過預訓練幫助我們加快訓練時間，從頭訓練一個模型需要大量時間、計算資源。幫助我們提升模型的擷取特徵的泛化能力
   

    """)

    st.markdown("""## 可用模型""")
    vgg16image = Image.open("availableModel.jpg")
    st.image("availableModel.jpg")
    st.markdown("""
    # VGG介紹
    - 2015年發表在ICLR(International Conference on Learning Representations)的會議論文
    - VGG是由Simonyan和Zisserman在文獻《Very Deep Convolutional Networks for Large Scale Image Recognition》中提出卷積神經網絡模型，其名稱來源於作者所在的牛津大學視覺幾何組(Visual Geometry Group)的縮寫。
    - 該模型參加2014年的ImageNet圖像分類與定位挑戰賽，取得了優異成績：在分類任務上排名第二，在定位任務上排名第一。
    
    """)

    st.markdown("""## VGG16模型架構圖""")
    vgg16image = Image.open("vgg16.png")
    st.image(vgg16image, caption='VGG16架構圖')
    st.markdown("""## Resnet50模型架構圖""")
    
    #resimage = Image.open("")
    st.image("resnet50.png", caption='Resnet50架構圖')
elif rad=="Source code":

    st.markdown("""
    ## Classify ImageNet classes with ResNet50
    ```python
    from tensorflow.keras.applications.resnet50 import ResNet50
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
    import numpy as np

    model = ResNet50(weights='imagenet')

    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    print('Predicted:', decode_predictions(preds, top=3)[0])
    # Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]
    ```
    
    """)
    st.markdown("""
    ## Fine-tune InceptionV3 on a new set of classes
    ```python
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(200, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    # train the model on the new data for a few epochs
    model.fit(...)

    # at this point, the top layers are well trained and we can start fine-tuning
    # convolutional layers from inception V3. We will freeze the bottom N layers
    # and train the remaining top layers.

    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    for i, layer in enumerate(base_model.layers):
    print(i, layer.name)

    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    for layer in model.layers[:249]:
    layer.trainable = False
    for layer in model.layers[249:]:
    layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    from tensorflow.keras.optimizers import SGD
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    model.fit(...)
    ```
    """)
elif rad=="結論":
    st.markdown("""
    ## 遇到問題?過程學到什麼
    Ans:  
    * 模型沒有train好。模型還有問題沒有改善  
    * streamlit圖片上傳串接model有問題，可能會出現OOM(out of memory)或者channel問題  
    * google爬蟲抓圖問題太多->改用kaggle提供的dataset
    * 目前model仍有問題，demo準確度與測試時不符
    * 學到了streamlit這個東西，對於一些資料科學前後端的及時交互很方便  
    
    
    """)

elif rad =="參考來源":
    
        
        
    st.markdown("""
    ### 參考來源
    streamlit: https://streamlit.io/  
    keras api: https://keras.io/api/applications/  
    kaggle: https://www.kaggle.com/  
    google search: https://github.com/Nv7-GitHub/googlesearch  
    https://zhuanlan.zhihu.com/p/350157158  
    https://blog.csdn.net/Castlehe/article/details/108973319  
    """)

if __name__ == '__main__':
   
    pass