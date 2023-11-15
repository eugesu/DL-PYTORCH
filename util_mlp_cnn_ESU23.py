# Funciones de utilidad para trabajar con MLP y CNN
# Eugenio Sánchez | Noviembre 2023
# IIT - ICAI - Comillas

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



def calcula_aciertos_modelo(loader, model, conj, device = torch.device('cpu')):
    ''' Calcula y muestra el nº de imágenes correctamente clasificadas, las 
        incorrectas y el porcentaje de aciertos
           - loader: cargador a utilizar
           - model: modelo a evaluar en el conjunto de datos del loader
           - conj: texto libre para indicar el nombre del fichero
    '''

    # desactiva cálculo gradientes
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for imgs, etqs in loader:
            imgs = imgs.reshape(-1, 28*28).to(device)
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


def calcula_matriz_confusion_modelo(loader, model, conj, num_classes, device = torch.device('cpu')):
    ''' Calcula y muestra la matriz de confusión
           - loader: cargador a utilizar
           - model: modelo a evaluar en el conjunto de datos del loader
           - conj: texto libre para indicar el nombre del fichero
    '''

    # matriz de confusión
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=float)
    with torch.no_grad():
        for i, (imgs, etqs) in enumerate(loader):
            imgs = imgs.reshape(-1, 28*28).to(device)
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
