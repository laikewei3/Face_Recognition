import face_recognition
import cv2
import numpy as np
import pickle

file = open("ref_name.pkl", "rb")
ref_dict = pickle.load(file)
file.close()

file = open("ref_embed.pkl", "rb")
embed_dict = pickle.load(file)
file.close()

known_face_encodings = []
known_face_names = []

for ref_id, embed_list in embed_dict.items():
    for my_embed in embed_list:
        known_face_encodings += [my_embed]
        known_face_names += [ref_id]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

webcam = cv2.VideoCapture(0)
while True:
    ret, frame = webcam.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)

    process_this_frame = not process_this_frame

    for (top_s, right, bottom, left), name in zip(face_locations, face_names):
        top_s *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top_s), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        if ref_dict.get(name):
            cv2.putText(frame, ref_dict[name], (left + 6, bottom - 6), font, 1, (255, 255, 255), 2, 1)
        else:
            cv2.putText(frame, "Unknown", (left + 6, bottom - 6), font, 1, (255, 255, 255), 2, 1)
        cv2.imshow('Video', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
