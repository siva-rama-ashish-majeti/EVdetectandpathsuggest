from flask import Flask, render_template, request
import os
import requests
from ultralytics import YOLO
import json
from geopy.distance import geodesic
import qrcode
import subprocess
import time
import cv2

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = YOLO("best.pt")

# Vehicle location (Bhimavaram default)
VEHICLE_LAT = 16.54310
VEHICLE_LON = 81.49629

ORS_API_KEY = "5b3ce3597851110001cf624862483d8b013343c688e1bac186fb9099"

# 🔁 Start ngrok
def start_ngrok():
    try:
        process = subprocess.Popen(["ngrok", "http", "5000"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(5)
        res = requests.get("http://127.0.0.1:4040/api/tunnels")
        tunnels = res.json()["tunnels"]
        for tunnel in tunnels:
            if tunnel["proto"] == "https":
                return tunnel["public_url"]
    except Exception as e:
        print("Ngrok startup error:", e)
    return None

# 🏥 Find closest hospital
def find_nearest_hospital(lat, lon):
    overpass_url = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json];
    (
      node["amenity"="hospital"](around:10000,{lat},{lon});
      way["amenity"="hospital"](around:10000,{lat},{lon});
      relation["amenity"="hospital"](around:10000,{lat},{lon});
    );
    out center;
    """
    try:
        response = requests.post(overpass_url, data=query)
        response.raise_for_status()
        data = response.json()
        min_dist = float('inf')
        nearest = None
        for element in data.get("elements", []):
            if "lat" in element and "lon" in element:
                coords = (element["lat"], element["lon"])
            elif "center" in element:
                coords = (element["center"]["lat"], element["center"]["lon"])
            else:
                continue
            dist = geodesic((lat, lon), coords).meters
            if dist < min_dist:
                min_dist = dist
                nearest = (
                    coords[0],
                    coords[1],
                    element.get("tags", {}).get("name", "Unnamed Hospital")
                )
        if nearest:
            print(f"✅ Nearest hospital: {nearest[2]} ({min_dist:.2f} meters away)")
            return nearest
    except Exception as e:
        print("Hospital fetch error:", e)
    return None, None, None

# 🚗 Route fetcher
def get_route_geojson(vehicle_coords, hospital_coords):
    url = "https://api.openrouteservice.org/v2/directions/driving-car/geojson"
    headers = {
        'Authorization': ORS_API_KEY,
        'Content-Type': 'application/json'
    }
    body = {
        "coordinates": [
            [vehicle_coords[1], vehicle_coords[0]],
            [hospital_coords[1], hospital_coords[0]]
        ]
    }
    try:
        res = requests.post(url, headers=headers, json=body)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        print("ORS Error:", e)
    return None

# 🧠 Main route
@app.route("/", methods=["GET", "POST"])
def index():
    image_path = None
    labels = None
    vehicle_coords = None
    hospital_coords = None
    hospital_name = None
    geojson_data = None
    qr_path = None

    if request.method == "POST":
        file = request.files["media"]
        if file:
            filename = file.filename
            ext = filename.split('.')[-1].lower()
            input_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(input_path)

            label_names = []

            if ext in ["jpg", "jpeg", "png"]:
                results = model(input_path)
                names = model.names
                labels = results[0].boxes.cls.tolist()
                label_names = list(set([names[int(cls)] for cls in labels])) if labels else ["No Detection"]
                image_path = input_path

            elif ext in ["mp4", "avi", "mov"]:
                results = model.predict(source=input_path, stream=True)
                names = model.names
                all_labels = []
                frame_saved = False

                for i, frame in enumerate(results):
                    frame_labels = frame.boxes.cls.tolist() if frame.boxes is not None else []
                    all_labels.extend([names[int(cls)] for cls in frame_labels])

                    if not frame_saved and frame.orig_img is not None:
                        preview_filename = f"preview_{os.path.splitext(filename)[0]}.png"
                        preview_path = os.path.join(UPLOAD_FOLDER, preview_filename)

                        # ✅ Correct: save BGR image directly without color conversion
                        cv2.imwrite(preview_path, frame.orig_img)
                        image_path = preview_path
                        frame_saved = True

                label_names = list(set(all_labels)) if all_labels else ["No Detection"]

            if "ambulance" in label_names:
                vehicle_coords = (VEHICLE_LAT, VEHICLE_LON)
                hospital_lat, hospital_lon, hospital_name = find_nearest_hospital(*vehicle_coords)
                if hospital_lat and hospital_lon:
                    hospital_coords = (hospital_lat, hospital_lon)
                    geojson_data = get_route_geojson(vehicle_coords, hospital_coords)

                    with open("static/route.json", "w") as f:
                        json.dump(geojson_data, f)

                    public_url = start_ngrok()
                    if public_url:
                        qr_url = f"{public_url}/viewroute"
                        qr = qrcode.make(qr_url)
                        qr_path = "static/qr.png"
                        qr.save(qr_path)
                    else:
                        print("⚠️ Ngrok failed. No QR will be generated.")

            return render_template("index.html",
                                   image_path=image_path,
                                   labels=label_names,
                                   vehicle_coords=vehicle_coords,
                                   hospital_coords=hospital_coords,
                                   hospital_name=hospital_name,
                                   route_geojson=json.dumps(geojson_data) if geojson_data else None,
                                   qr_path=qr_path)

    return render_template("index.html",
                           image_path=None,
                           labels=None,
                           vehicle_coords=None,
                           hospital_coords=None,
                           hospital_name=None,
                           route_geojson=None,
                           qr_path=None)

# 🌐 Route viewer
@app.route("/viewroute")
def viewroute():
    try:
        with open("static/route.json") as f:
            geojson_data = json.load(f)
        return render_template("viewroute.html",
                               vehicle_coords=(VEHICLE_LAT, VEHICLE_LON),
                               route_geojson=json.dumps(geojson_data))
    except:
        return "No route found."

if __name__ == "__main__":
    app.run(port=5000, debug=True)
