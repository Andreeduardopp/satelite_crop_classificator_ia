import requests
from .load_balancer import LoadBalancer
from focusnet.models import TaskStatusEnum, LogsTask,Task
from focusnet import database,models
import asyncio
import numpy as np
from PIL import Image
#from tensorflow.keras.preprocessing import image
import base64
import time

from .utils import load_buffer

def normaliza(valor, minimo, maximo):
    """
    Normaliza um valor para o intervalo [0, 1].

    Argumentos:
        valor (float): O valor a ser normalizado.
        minimo (float): O valor mínimo no intervalo original.
        maximo (float): O valor máximo no intervalo original.

    Retorna:
        float: O valor normalizado no intervalo [0, 1].
    """
    return(valor - minimo)/(maximo - minimo)


def request_mlserver(endpoint: str, caminho_imagem: str, mes: int, container_id):
    print("Requesting MLServer")
    imagem = caminho_imagem
    tamanho_default = (224, 224)

    file = {'imagem': (imagem, open(imagem, 'rb'))}
    tentativas = 0

    try:
        # Primeiro escopo de sessão para buscar informações do container
        with database.SessionLocal() as db:
            container = db.query(models.ContainerStatus).filter(models.ContainerStatus.id == container_id).first()
            if not container:
                raise ValueError(f"Container com ID {container_id} não encontrado.")
            log = models.LogsTask(task_id=container.task_id)

        # Construção da URL e normalização do mês
        url = f'{container.url}:{container.porta}{endpoint}'
        mes_normalizado = normaliza(mes, 1, 12)
        data = {"mes": mes_normalizado}

        try:
            response = requests.post(url, data=data, files=file)
            response.raise_for_status()
            #time.sleep(10)
            # Segundo escopo de sessão para atualizar estados no banco
            with database.SessionLocal() as db:
                container = db.query(models.ContainerStatus).filter(models.ContainerStatus.id == container_id).first()
                container.set_idle(db)
                log.complete_task(db)
                db.commit()
                db.refresh(container)
            return response
        except requests.exceptions.HTTPError as e:
            mensagem = f"Erro HTTP ocorreu: {e}"
            with database.SessionLocal() as db:
                container.set_error(db)
                log.fail_task(db, mensagem_erro=mensagem, container_id=container.id)
                db.commit()
                db.refresh(container)
            print(mensagem)
        except Exception as e:
            mensagem = f"Erro: {e}"
            with database.SessionLocal() as db:
                container.set_error(db)
                log.fail_task(db, mensagem_erro=mensagem, container_id=container.id)
                db.commit()
                db.refresh(container)
            print(mensagem)
    except Exception as e:
        print(f"Erro geral: {e}")

    return {'status': 'ERRO', 'mensagem': "Máximo de tentativas"}

