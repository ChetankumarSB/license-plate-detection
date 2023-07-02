from flask import Flask, request, jsonify
import cv2
import numpy as np
import imutils
import easyocr
import re
import os

app = Flask(__name__)

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No image file uploaded'})

        image = request.files['file']
        image_path = 'uploaded_image.png'
        image.save(image_path)

        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bfilter = cv2.bilateralFilter(gray, 11, 11, 17)
        edged = cv2.Canny(bfilter, 30, 200)

        contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        location = None
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            if len(approx) == 4:
                location = approx
                break

        if location is not None:
            mask = np.zeros(gray.shape, np.uint8)
            new_image = cv2.drawContours(mask, [location], 0, 255, -1)
            new_image = cv2.bitwise_and(img, img, mask=mask)

            (x, y) = np.where(mask == 255)
            (x1, y1) = (np.min(x), np.min(y))
            (x2, y2) = (np.max(x), np.max(y))
            cropped_image = gray[x1:x2 + 3, y1:y2 + 3]

            reader = easyocr.Reader(['en'])
            result = reader.readtext(cropped_image)

            result_text = ""
            for detection in result:
                if detection[1] == "IND":
                    continue
                result_text += detection[1].replace(" ", "").upper()

            result_text = re.sub(r'[^a-zA-Z0-9]', '', result_text)

            if len(result_text) == 10:
                return jsonify({'number_plate': result_text})
            else:
                return jsonify({'error': 'Unable to detect valid number plate text'})
        else:
            return jsonify({'error': 'No contour with 4 sides found'})
    except Exception as e:
        return jsonify({'error': str(e)})
    finally:
        os.remove(image_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003)
