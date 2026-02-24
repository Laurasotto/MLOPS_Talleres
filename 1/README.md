#Realizado por Laura Sotto y Thomas Rivera

1. Definimos 2 servicios en services: Jupyter y API)

<img width="1142" height="591" alt="image" src="https://github.com/user-attachments/assets/7dfc7788-ade1-4e2f-968f-ca709fd6eaba" />

2. Definir un volumen nombrado en volumes: y montarlo en ambos servicios para compartir archivos. Para Jupyter, Docker tiene guía específica de “Data science with JupyterLab”
  - services:
  jupyter:
    image: jupyter-image
    ports:
      - "8888:8888"
    volumes:
      - shared_models:/shared
    environment:
      - MODEL_DIR=/shared/model

  - api:
    image: api-image
    ports:
      - "8989:8989"
    volumes:
      - shared_models:/shared
    environment:
      - MODEL_DIR=/shared/model
  
De esta manera lo monta dentro de cada contenedor y así ambos comparten archivos.
3. Luego creamos las dos imagenes de Dockerfile.

<img width="388" height="114" alt="image" src="https://github.com/user-attachments/assets/1486c013-0c2c-46f5-93ab-16c77c6a9f4b" />

4. Luego modificamos el codigo de para que use el volumen compartido en el modelo1.py que es donde entrenamos los tres metodos y la api.py.
<img width="1478" height="436" alt="image" src="https://github.com/user-attachments/assets/dfccf2cb-53a1-4d73-b0de-d878c6297e03" />

5. Levantar todo con docker compose up --build
<img width="1280" height="801" alt="image" src="https://github.com/user-attachments/assets/83b5fa68-2446-4965-aa6f-6de9b1c86f6b" />

6. Abrimos Jupyter http://localhost:8888, con el token que se despliega en la terminal local ingresamos. 
<img width="1304" height="1600" alt="image" src="https://github.com/user-attachments/assets/96239d09-21d6-4490-bfe3-bcacfa362b19" />

7. En Jupyter corremos el modelo,y lo entrenamos.
<img width="1426" height="616" alt="image" src="https://github.com/user-attachments/assets/edfce789-6ea9-4561-9fb5-a41b178f4772" />

8. Luego verificamos que los modelos esten guardados en el volumen compartido.
<img width="506" height="87" alt="image" src="https://github.com/user-attachments/assets/15d415a3-1c2b-4b14-8341-c36c64ad2fa2" />

9. Luego verificamos en la API http://localhost:8989/docs y realizamos distintos testeos para ver si los modelos estan corriendo correctamente en la API

<img width="1600" height="764" alt="image" src="https://github.com/user-attachments/assets/5d3b2db6-8b51-4d89-9e41-0f5dacc62e28" />

<img width="808" height="1386" alt="image" src="https://github.com/user-attachments/assets/4192c32f-8a4a-4a60-8b8c-7cfde1e16f31" />

 

