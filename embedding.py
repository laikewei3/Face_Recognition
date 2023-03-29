import face_recognition
import cv2
import pickle

ref_id = input("Enter id: ")
name = input("Enter the name: ")

try:
    file = open("ref_name.pkl", "rb")
    ref_dict = pickle.load(file)
    file.close()
except:
    ref_dict = {}

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

for i in range(5):
    key = cv2.waitKey(1)
    webcam = cv2.VideoCapture(0)
    while True:
        check, frame = webcam.read()
        cv2.imshow("Capturing",frame)
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

file = open("ref_embed.pkl", "wb")
pickle.dump(embed_dict, file)
file.close()
