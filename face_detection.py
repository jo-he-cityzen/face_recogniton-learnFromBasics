# importing librarys
import cv2
import numpy as npy
import face_recognition as face_rec
# function
def resize(img, size) :
    width = int(img.shape[1]*size)
    height = int(img.shape[0] * size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation= cv2.INTER_AREA)


# img declaration
joshik = face_rec.load_image_file('sampe_images\joshik.jpg')
joshik = cv2.cvtColor(joshik, cv2.COLOR_BGR2RGB)
joshik = resize(joshik, 0.50)
joshik_test = face_rec.load_image_file('sampe_images\elonmusk.jpg')
joshik_test = resize(joshik_test, 0.50)
joshik_test = cv2.cvtColor(joshik_test, cv2.COLOR_BGR2RGB)

# finding face location

faceLocation_joshik = face_rec.face_locations(joshik)[0]
encode_joshik = face_rec.face_encodings(joshik)[0]
cv2.rectangle(joshik, (faceLocation_joshik[3], faceLocation_joshik[0]), (faceLocation_joshik[1], faceLocation_joshik[2]), (255, 0, 255), 3)


faceLocation_joshiktest = face_rec.face_locations(joshik_test)[0]
encode_joshiktest = face_rec.face_encodings(joshik_test)[0]
cv2.rectangle(joshik_test, (faceLocation_joshik[3], faceLocation_joshik[0]), (faceLocation_joshik[1], faceLocation_joshik[2]), (255, 0, 255), 3)

results = face_rec.compare_faces([encode_joshik], encode_joshiktest)
print(results)
cv2.putText(joshik_test, f'{results}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255), 2 )

cv2.imshow('main_img', joshik)
cv2.imshow('test_img', joshik_test)
cv2.waitKey(0)
cv2.destroyAllWindows()