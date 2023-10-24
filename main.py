from fastapi import FastAPI, HTTPException, Response
from starlette.status import HTTP_200_OK, HTTP_201_CREATED, HTTP_204_NO_CONTENT
from model.face_connection import lineConnection
from schema.face_schema import InputSchema

app = FastAPI()
conn = lineConnection()

@app.post("/api/entradas/face_recognition/insert", status_code=HTTP_201_CREATED)
def insert_entrada(entrada_data: InputSchema):
    data = entrada_data.dict()
    conn.write_entrada(data)
    return Response(status_code=HTTP_201_CREATED)

@app.get("/api/entradas/face_recognition/{id}", status_code=HTTP_200_OK)
def get_one_entrada(id: int):
    data = conn.read_one_entrada(id)
    if data:
        entrada = {
            "id": data[0],
            "led_red": data[1],
            "led_green": data[2],
            "led_mirrow": data[3]
        }
        return entrada
    else:
        raise HTTPException(status_code='HTTP_404_NOT_FOUND', detail="Entrada no encontrado")
        
@app.get("/api/entradas/allface_recognition/", status_code=  HTTP_200_OK)
def allEntradas():
    
    items = []
    
    for data in conn.read_all_entradas():
        dictionary = {}
        dictionary["id"] = data[0]
        dictionary["led_red"] = data[1]
        dictionary["led_green"] = data[2]
        dictionary["led_mirrow"] = data[3]
        items.append(dictionary)
    
    return items 

@app.put("/api/entradas/face_recognition/update/{id}", status_code=HTTP_204_NO_CONTENT)
def update(entrada_data: InputSchema, id:str):
    data = entrada_data.dict()
    data["id"] = id
    
    conn.update_entrada(data) 
    return Response(status_code=HTTP_204_NO_CONTENT) 


# Agrega las rutas y controladores para otras operaciones CRUD (update y delete) según sea necesario.

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)    