import face_recognition
import cv2
import pickle
import filetype
import urllib.request
from urllib.error import HTTPError

ref_id = input("Enter id: ")

try:
    file = open("ref_name.pkl", "rb")
    ref_dict = pickle.load(file)
    file.close()
except:
    ref_dict = {}

if ref_id in ref_dict:
    name = ref_dict.get(ref_id)
    print("ID:",ref_id,"Name:",name)
    change = input("Do u want to change the name? (Enter 'y' to change the name)")
    if change == 'y':
        name = input("Enter the name: ")
else:
    name = input("Enter the name: ")

ref_dict[ref_id] = name

file = open("ref_name.pkl", "wb")
pickle.dump(ref_dict, file)
file.close()

try:
    file = open("ref_embed.pkl", "rb")
    embed_dict = pickle.load(file)
    file.close()
except:
    embed_dict = {}

condition = -1
while condition != 0 and condition != 1 and condition != 2:
    try:
        condition = int(input("Enter 0 for camera, 1 for local image, 2 for links: "))
    except ValueError:
        print("This is not number, insert only 0,1 or 2.")

if condition == 0:
    for i in range(5):
        key = cv2.waitKey(1)
        webcam = cv2.VideoCapture(0)
        while True:
            check, frame = webcam.read()
            cv2.imshow("Capturing", frame)
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_same_frame = small_frame[:, :, ::-1]

            key = cv2.waitKey(1)

            if key == ord('s'):
                face_locations = face_recognition.face_locations(rgb_same_frame)
                if face_locations != []:
                    face_encoding = face_recognition.face_encodings(frame)[0]
                    if ref_id in embed_dict:
                        embed_dict[ref_id] += [face_encoding]
                    else:
                        embed_dict[ref_id] = [face_encoding]
                print("Captured")
                webcam.release()
                cv2.destroyAllWindows()
                break
            elif key == ord('q'):
                print("Turning off camera.")
                webcam.release()
                print("Camera off.")
                print("Program ended")
                cv2.destroyAllWindows()
                break
elif condition == 1:
    while True:
        while True:
            path = input("Enter your image path:")
            while not filetype.is_image(path):
                print("Invalid Image Path, Please enter a valid image path.")
                path = input("Enter your image path:")
            image = face_recognition.load_image_file(path)
            face_location = face_recognition.face_locations(image)
            if len(face_location) != 1:
                print("The picture should has one and only one person")
            else:
                break
        face_encoding = face_recognition.face_encodings(image)[0]
        if ref_id in embed_dict:
            embed_dict[ref_id] += [face_encoding]
        else:
            embed_dict[ref_id] = [face_encoding]
        continue_embed = input("Do you have another image?('y' for yes)")
        if continue_embed != 'y':
            break
elif condition == 2:
    while True:
        while True:
            while True:
                url = input("Enter your image url:")
                try:
                    request = urllib.request.urlopen(url)
                    break
                except HTTPError:
                    print("Sorry, we cannot access to this link, please change other links")
            image = face_recognition.load_image_file(request)
            face_location = face_recognition.face_locations(image)
            if len(face_location) != 1:
                print("The picture should has one and only one person")
            else:
                break
        face_encoding = face_recognition.face_encodings(image)[0]
        if ref_id in embed_dict:
            embed_dict[ref_id] += [face_encoding]
        else:
            embed_dict[ref_id] = [face_encoding]
        continue_embed = input("Do you have another image?('y' for yes)")
        if continue_embed != 'y':
            break

file = open("ref_embed.pkl", "wb")
pickle.dump(embed_dict, file)
file.close()
