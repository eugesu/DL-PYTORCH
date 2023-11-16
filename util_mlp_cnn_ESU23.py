# Funciones de utilidad para trabajar con MLP y CNN
# Eugenio Sánchez | Noviembre 2023
# IIT | Comillas - ICAI

# importa
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import idx2numpy

# clase para crear los datasets
class MiDataset(Dataset):
    def __init__(self, imagefile, labelfile, transform = transforms.ToTensor()):
        self.imagearray = idx2numpy.convert_from_file(imagefile)
        self.labelarray = idx2numpy.convert_from_file(labelfile)
        self.transform = transform  

    def __len__(self):
        return len(self.labelarray)

    def __getitem__(self, idx):
        image = self.imagearray[idx]
        label = self.labelarray[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def calcula_aciertos_modelo(loader, model, conj, hacer_reshape = True, device = torch.device('cpu')):
    ''' Calcula y muestra el nº de imágenes correctamente clasificadas, las 
        incorrectas y el porcentaje de aciertos
           - loader: cargador a utilizar
           - model: modelo a evaluar en el conjunto de datos del loader
           - conj: texto libre para indicar el nombre del fichero
           - hacer_reshape: a True para el caso del MLP
    '''

    # desactiva cálculo gradientes
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for imgs, etqs in loader:
            if hacer_reshape:
                imgs = imgs.reshape(-1, 28*28)
            imgs = imgs.to(device)
            etqs = etqs.to(device)  
            outputs = model(imgs)

            _, predictions = torch.max(outputs, 1)
            n_samples += etqs.shape[0] 
            n_correct += (predictions == etqs).sum().item()

        acc = 100.0 * n_correct / n_samples
        n_incorrect = n_samples - n_correct

        print(f'{conj} ----------')
        print(f'   Correctas ({conj}):{n_correct} Incorrectas({conj}):{n_incorrect}')
        print(f'   Aciertos ({conj}): {round(acc,3)} %')


def calcula_matriz_confusion_modelo(loader, model, conj, num_classes, hacer_reshape = True, device = torch.device('cpu')):
    ''' Calcula y muestra la matriz de confusión
           - loader: cargador a utilizar
           - model: modelo a evaluar en el conjunto de datos del loader
           - conj: texto libre para indicar el nombre del fichero
           - hacer_reshape: a True para el caso del MLP
    '''

    # matriz de confusión
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=float)
    with torch.no_grad():
        for i, (imgs, etqs) in enumerate(loader):
            if hacer_reshape:
                imgs = imgs.reshape(-1, 28*28)
            imgs = imgs.to(device)
            etqs = etqs.to(device)  
            outputs = model(imgs)
            _, predictions = torch.max(outputs, 1)

            for t, p in zip(etqs.view(-1), predictions.view(-1)):
                confusion_matrix[p.long(), t.long()] += 1

    # eliminamos los aciertos para facilitar el foco en los errores de clasificación
    confusion_matrix[confusion_matrix>100] = float("nan")

    # representa fallos
    fig = plt.figure(1,figsize=(8, 2))
    plt.subplot(1,2,1)
    plt.bar(range(10), np.nansum(confusion_matrix.numpy(), axis=0))
    plt.xticks(range(10))
    plt.grid()
    plt.title('Número de errores según dígito')
    plt.xlabel('Dígito real')
    plt.ylabel(f'Nº de confusiones en {conj}')

    plt.subplot(1,2,2)
    plt.barh(range(10), np.nansum(confusion_matrix.numpy(), axis=1))
    plt.yticks(range(10))
    plt.grid()
    plt.title('Número de errores según dígito estimado')
    plt.ylabel('Dígito estimado')
    plt.xlabel(f'Nº de confusiones en {conj}')
    plt.show()
    
    # representa la matriz de confusión
    fig = plt.figure(2,figsize=(8, 3))
    sns.heatmap(confusion_matrix, annot=True, linewidth=.5,  cmap="crest")
    plt.title(f'Matriz de confusión en {conj} (solo fallos)')
    plt.xlabel('Dígito real')
    plt.ylabel('Dígito estimado')
    plt.show()



def muestra_ejemplos_fallos(loader, model, conj, digito = 5, hacer_reshape = True, device = torch.device('cpu')):
    ''' Determina y muestra ejemplos de imágenes mal clasificadas
           - loader: cargador a utilizar
           - model: modelo a evaluar en el conjunto de datos del loader
           - conj: texto libre para indicar el nombre del fichero
           - digito: el dígito específico que se quiere ver sus fallos
           - hacer_reshape: a True para el caso del MLP
    '''
    
    # guarda en un nuevo tensor todos los casos incorrectos
    if hacer_reshape:
        err_img = torch.empty(1, 28*28, dtype=float)
    else:
        err_img = torch.empty(1, 1, 28, 28, dtype=float)
    
    err_etq_real = torch.empty(1, 1, dtype=torch.uint8)
    err_etq_est = torch.empty(1, 1, dtype=torch.uint8)

    with torch.no_grad():
        for i, (imgs, etqs) in enumerate(loader):
            if hacer_reshape:
                imgs = imgs.reshape(-1, 28*28)
            imgs = imgs.to(device)
            etqs = etqs.to(device)  
            outputs = model(imgs)
            _, predictions = torch.max(outputs, 1)

            # coge los fallos
            ind_fallos = etqs != predictions

            err_img = torch.cat((err_img, imgs[ind_fallos]), dim=0)
            err_etq_real = torch.cat((err_etq_real, etqs[ind_fallos].unsqueeze(0)), dim=1)
            err_etq_est = torch.cat((err_etq_est, predictions[ind_fallos].unsqueeze(0)), dim=1)

    # pinta unos cuantos fallos
    print(f'Muestra 100 ejemplos de imágenes mal clasificadas en {conj}:')
    fig = plt.figure(1,figsize=(18, 18))
    k=1
    i=0
    while k<100 and i < len(err_etq_real[0,:])-1:
        plt.subplot(10,10,k)
        plt.imshow(err_img[i+1].reshape(28,28), cmap=plt.cm.binary)
        plt.xticks([])
        plt.yticks([])
        plt.title(f'REAL: {err_etq_real[0,i+1].item()} EST:{err_etq_est[0,i+1].item()}')
        k = k+1
        i=i+1

    plt.show()
    
    # pinta unos cuantos fallos
    print(f'Muestra 20 ejemplos de imágenes mal clasificadas en {conj} del dígito {digito}:')
    fig = plt.figure(2,figsize=(15, 3))
    k=1
    i=0
    while k<20 and i < len(err_etq_real[0,:])-1:
        if err_etq_real[0,i+1].item() == digito:
            plt.subplot(2,10,k)
            plt.imshow(err_img[i+1].reshape(28,28), cmap=plt.cm.binary)
            plt.xticks([])
            plt.yticks([])
            plt.title(f'REAL: {err_etq_real[0,i+1].item()} EST:{err_etq_est[0,i+1].item()}')
            k = k+1

        i=i+1

    plt.show()

