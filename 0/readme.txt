#realizado por laura sotto y thomas rivera

#clonar el repositorio

$git clone https://github.com/Laurasotto/MLOPS_Talleres.git


#instalar los prerequisitos
$python3 -m pip install -r requistos.txt

#pre entrenar el modelo para usarlo despues
$python3 modelo1.py

#levantar el api 
$python3 -m uvicorn api:app --host 0.0.0.0 --port 8989

#abrir en el navegador, copia y pega directamente en el navegador
http://localhost:8989/docs

#construir la imagen del docker 
$docker build --no-cache -t penguins-api .

#ejecutar y encender el contenedor
docker run --rm -p 8989:8989 penguins-api

#dentro del link en internet, puedes hacer lo siguiente
#verificar tipos de modelos disponibles 
GET /models
#respuesta esperada
{
  "available": ["dt", "lr", "rf"],
  "active": "rf"
}

#cambiar modelo
POST /models/select
#enviar
{
  "name": "dt"
}
#respuesta

{
  "active": "dt"
}

#para realiar una prediccion
POST /predict
#enviar este formato, cambiar datos

{
  "bill_length_mm": 39.1,
  "bill_depth_mm": 18.7,
  "flipper_length_mm": 181,
  "body_mass_g": 3750,
  "sex": "male",
  "island": "Torgersen",
  "year": 2007
}

#respuesta para el ejemplo especificado anteriormente 
{
  "prediction": "Adelie",
  "model_used": "dt"
}
