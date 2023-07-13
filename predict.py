from tensorflow.keras.models import load_model
import os
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

classes = os.listdir('./train')
model = load_model("leaf_disease.h5")
label_dic = {
      "うどんこ病":0,
      "もち病":1,
      "ベト病":2,
      "モザイク病":3,
      "正常":4,
      "炭疽病":5,
      "白サビ病":6,
      "角斑病":7,
      "赤星病":8,
      "黒星病":9,
      "黒枯病":10
}


def get_label(path):
  if "udonnkobyou" in path:
    return "うどんこ病"
  elif "mochibyou" in path:
    return "もち病"
  elif "betobyou" in path:
    return "ベト病"
  elif "mozaikubyou" in path:
    return "モザイク病"
  elif "nomal" in path:
    return "正常"
  elif "tannsobyou" in path:
    return "炭疽病"
  elif "shirosabibyou" in path:
    return "白サビ病"
  elif "kakuhannbyou" in path:
    return "角斑病"
  elif "akaboshibyou" in path:
    return "赤星病"
  elif "kuroboshibyou" in path:
    return "黒星病"
  elif "kurogarebyou" in path:
    return "黒枯病"
  else:
    pass

# 一問一答
# root_path = "./検証"
# test_img_list = os.listdir(root_path)
# true_num = 0
# false_num = 0
# for path in test_img_list:
#   input = image.load_img(os.path.join(root_path,path),target_size=(256,256))
#   input = np.expand_dims(input,axis=0)
#   input = preprocess_input(input)
#   result = model.predict(input)
#   predict_label = list(label_dic.keys())[np.argmax(result[0])]
#   acc_label = get_label(path)
#   print(predict_label, acc_label, path)
#   # 正答率計算
#   if predict_label == acc_label:
#     true_num += 1

# print("正答率は{}".format(true_num/(len(test_img_list))))

# 3択問題に予測したやつが入っているか
# root_path = "./検証"
# test_img_list = os.listdir(root_path)
# true_num = 0
# false_num = 0
# top_k = 3
# for path in test_img_list:
#   input = image.load_img(os.path.join(root_path,path),target_size=(256,256))
#   input = np.expand_dims(input,axis=0)
#   input = preprocess_input(input)
#   result = model.predict(input)
#   predict_label = [list(label_dic.keys())[index] for index in np.argsort(result[0])[::-1][:top_k]]
#   acc_label = get_label(path)
#   print(predict_label, acc_label, path)
#   if acc_label in predict_label:
#     true_num +=1

# print("正答率は{}".format(true_num/(len(test_img_list))))

root_path = "./検証"
test_img_list = os.listdir(root_path)
true_num = 0
false_num = 0
top_k = 3
for path in test_img_list:
  input = image.load_img(os.path.join(root_path,path),target_size=(256,256))
  input = np.expand_dims(input,axis=0)
  input = preprocess_input(input)
  result = model.predict(input)
  predict_label = {list(label_dic.keys())[index]: result[0][index] for index in np.argsort(result[0])[::-1][:top_k]}
  acc_label = get_label(path)
  print(predict_label, acc_label, path)
print("正答率は{}".format(true_num/(len(test_img_list))))
