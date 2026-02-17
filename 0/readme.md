#realizado por laura sotto y thomas rivera

#clonar el repositorio

$git clone https://github.com/Laurasotto/MLOPS_Talleres.git


#instalar los prerequisitos
$python3 -m pip install -r requistos.txt
<img width="1409" height="733" alt="image" src="https://github.com/user-attachments/assets/1ca09b82-2a14-420e-8be8-6557a10f05f0" />

#pre entrenar el modelo para usarlo despues
$python3 modelo1.py

#levantar el api 
$python3 -m uvicorn api:app --host 0.0.0.0 --port 8989
<img width="859" height="340" alt="image" src="https://github.com/user-attachments/assets/fa6fa20e-e6c0-4f51-9f88-99f15a8e566d" />

#abrir en el navegador, copia y pega directamente en el navegador
http://localhost:8989/docs
<img width="898" height="836" alt="image" src="https://github.com/user-attachments/assets/f55aec7a-a184-435a-b3d1-00b9670c9216" />

#construir la imagen del docker 
$docker build --no-cache -t penguins-api .
<img width="1698" height="906" alt="image" src="https://github.com/user-attachments/assets/31fa425f-b079-4ff6-8ac7-1a0b954bead5" />

#ejecutar y encender el contenedor
docker run --rm -p 8989:8989 penguins-api

#dentro del link en internet, puedes hacer lo siguiente
#verificar tipos de modelos disponibles 
<img width="1446" height="706" alt="image" src="https://github.com/user-attachments/assets/d516b880-6ab7-4ef2-a1a8-e3f7fbac8260" />

GET /models
#respuesta esperada
{
  "available": ["dt", "lr", "rf"],
  "active": "rf"
}

#cambiar modelo
<img width="692" height="370" alt="image" src="https://github.com/user-attachments/assets/91298ac1-2015-4887-8557-c3f249a0d231" />

POST /models/select
#enviar
{
  "name": "dt"
}
#respuesta
<img width="703" height="775" alt="image" src="https://github.com/user-attachments/assets/bbeb2dca-05c6-4e75-a4ff-9306a2e99c71" />

{
  "active": "dt"
}

#para realiar una prediccion
POST /predict
#enviar este formato, cambiar datos
<img width="470" height="422" alt="image" src="https://github.com/user-attachments/assets/69e2f759-4395-4b4e-a67c-e76dfee73edb" />

{
  "bill_length_mm": 39.1,
  "bill_depth_mm": 18.7,
  "flipper_length_mm": 181,
  "body_mass_g": 3750,
  "sex": "male",
  "island": "Torgersen",
  "year": 2007
}
<img width="391" height="110" alt="image" src="https://github.com/user-attachments/assets/a4bcb6f4-a00e-4a68-8ad5-094d1b61867a" />

#respuesta para el ejemplo especificado anteriormente 
{
  "prediction": "Adelie",
  "model_used": "dt"
}
