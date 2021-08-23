import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def display_scree_plot(pca):
        
    plt.figure(figsize = (9, 6))
    
    scree = pca.explained_variance_ratio_*100
    
    plt.bar(np.arange(len(scree)) + 1, scree)
    plt.plot(np.arange(len(scree)) + 1, scree.cumsum(), c = "red", marker = 'o')
    
    for x,y in zip(np.arange(len(scree)) + 1, scree.round(1)):
        plt.text(x, y, y, ha = 'center', va = 'bottom', fontsize = 15)
    
    plt.tick_params(axis = 'both', length = 0)
    plt.xticks(range(len(scree) + 1))
    
    plt.xlabel("Composantes principales")
    plt.ylabel("Pourcentage d'inertie")
    
    plt.title("Perte d'inertie")
    
    plt.show(block = False)

def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    for d1, d2 in axis_ranks: # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:

            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(6, 6))

            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="blue")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=0.7, color='blue'))
            
            # affichage des noms des variables  
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='15', ha='center', va='center', rotation= label_rotation, color="purple", alpha=1)
            
            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor = 'none', edgecolor = 'black')
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
            
            plt.yticks([])
            plt.xticks([])
        
            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)
        
def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None, lims=None):
    for d1,d2 in axis_ranks:
        if d2 < n_comp:
            
            # initialisation de la figure       
            fig = plt.figure(figsize=(10, 10))
            
            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
                plt.xlim(xmin * 1.1, xmax * 1.1)
                plt.ylim(ymin * 1.1, ymax * 1.1)
            else :
                boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
                xmin, xmax, ymin, ymax = -boundary, boundary, -boundary, boundary
                plt.xlim([-boundary,boundary])
                plt.ylim([-boundary,boundary])
                
            # affichage des points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value)
                plt.legend(loc = 'best', fontsize = 'x-small')

            # affichage des labels des points
            if labels is not None:  
                for i,(x, y) in enumerate(X_projected[:,[d1,d2]]):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='10', ha='center', va='bottom')
                
            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)

