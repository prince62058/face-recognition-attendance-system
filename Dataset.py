import cv2
import pickle
import numpy as np
import os


# Function to delete face data and name
def delete_face(name_to_delete):
    if 'names.pkl' in os.listdir('data/') and 'faces_data.pkl' in os.listdir('data/'):
        with open('data/names.pkl', 'rb') as f:
            names = pickle.load(f)

        with open('data/faces_data.pkl', 'rb') as f:
            faces_data = pickle.load(f)

        if name_to_delete in names:
            # Find all the indexes of the name to delete
            indexes_to_delete = [i for i, name in enumerate(names) if name == name_to_delete]

            # Delete corresponding face data
            names = [name for i, name in enumerate(names) if i not in indexes_to_delete]
            faces_data = np.delete(faces_data, indexes_to_delete, axis=0)

            # Save updated files
            with open('data/names.pkl', 'wb') as f:
                pickle.dump(names, f)

            with open('data/faces_data.pkl', 'wb') as f:
                pickle.dump(faces_data, f)

            print(f"Deleted all faces associated with {name_to_delete}")
        else:
            print(f"No faces found for {name_to_delete}")


# Function to collect face data
def collect_faces():
    video = cv2.VideoCapture(0)
    facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    faces_data = []
    i = 0
    name = input("Enter Your Name: ")

    while True:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            crop_img = frame[y:y + h, x:x + w, :]
            resized_img = cv2.resize(crop_img, (50, 50))
            if len(faces_data) <= 100 and i % 10 == 0:
                faces_data.append(resized_img)
            i += 1
            cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

        cv2.imshow("Frame", frame)
        k = cv2.waitKey(1)

        if len(faces_data) == 50:
            break

    video.release()
    cv2.destroyAllWindows()

    faces_data = np.asarray(faces_data)
    faces_data = faces_data.reshape(100, -1)

    # Load or create names list
    if 'names.pkl' not in os.listdir('data/'):
        names = [name] * 100
        with open('data/names.pkl', 'wb') as f:
            pickle.dump(names, f)
    else:
        with open('data/names.pkl', 'rb') as f:
            names = pickle.load(f)
        names = names + [name] * 100
        with open('data/names.pkl', 'wb') as f:
            pickle.dump(names, f)

    # Load or append faces data
    if 'faces_data.pkl' not in os.listdir('data/'):
        with open('data/faces_data.pkl', 'wb') as f:
            pickle.dump(faces_data, f)
    else:
        with open('data/faces_data.pkl', 'rb') as f:
            faces = pickle.load(f)
        faces = np.append(faces, faces_data, axis=0)
        with open('data/faces_data.pkl', 'wb') as f:
            pickle.dump(faces, f)


# Menu for user interaction
def main():
    while True:
        print("\n1. Collect face data")
        print("2. Delete a face by name")
        print("3. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            collect_faces()
        elif choice == '2':
            name_to_delete = input("Enter the name whose face data you want to delete: ")
            delete_face(name_to_delete)
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == '__main__':
    main()
