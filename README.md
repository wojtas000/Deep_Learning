Plan pracy

DONE na końcu - task skonczone
* - aktulnie w pracy

# Mikołaj
- research na temat sieci DONE
    - wstepnie wiem co i jak DONE

- Klasę sieci 
    - zaczynam to robic DONE
    - zrobiłem 3 sieci - zwykla siec bez warstw konwolucyjnych i dwie na podstawie artykułów - parametry warstw do ogarnięcia ale struktura jest zgodna z teorią DONE
    - klasa sieci dostosowana do zmian hiperparametrow DONE
    - master klasa do dziedziczenia 
    * - pretrained model Alexnet

- algorytm uczenia forward-backward DONE
    - zaimplementowane od razu w klasie DONE

- do ogarnięcia połaczenie klasy danych z modelem DONE

# Janek
- augmentacją danych 

- przygotowaniem hiperparametrów / algorytmu szukania hiperparametrów


# TASKI z polecenia

•Test and compare different network architecture (at least one should be aconvolutional neural network)
    - mamy 2 konwolucyjne i jedną zwykłą sieć

•Investigate influence of the following hyper-parameter change on obtained results:
    •At least 2 hyper-parameters related to training process
        - no of epochs?
        - parameter in optimizer?
    •At least 2 hyper-parameters related to regularization
        - pool size?
        - ilosc warstw konwolucyjnych?

•Investigate influence of at least X data augmentation techniques from the followinggroups:
    •Standard operations (where x=3)
    •More advanced data augmentation techniques like mixup, cutmix, cutout (where x=1)

•Consider application of ensemble (hard/soft voting, stacking)


FYI czym jest ensamble:
Ensemble learning (zespół metod uczących) to technika uczenia maszynowego polegająca na łączeniu wyników kilku modeli (np. drzew decyzyjnych, sieci neuronowych) w celu poprawienia ogólnej skuteczności predykcji.

Idea polega na tym, że kilka słabszych modeli połączonych razem może stworzyć silniejszy model. W ramach tej techniki stosuje się różne metody łączenia modeli, takie jak:

Bagging - polega na losowym wyborze podzbiorów danych ze zbioru treningowego i uczeniu kilku niezależnych modeli na tych podzbiorach. Następnie wyniki tych modeli są łączone w celu uzyskania ogólnego wyniku. Ta metoda zmniejsza wariancję wyników i poprawia stabilność predykcji.

Boosting - polega na tworzeniu sekwencji słabszych modeli, które są zbudowane na podstawie poprzednich modeli. Wagi próbek, które zostały źle sklasyfikowane przez poprzednie modele, są zwiększane, aby następny model skupił się na tych próbkach. Ta metoda zmniejsza błąd predykcji i zwiększa skuteczność klasyfikacji.

Stacking - polega na zastosowaniu kilku modeli, a następnie nauczeniu modelu meta-klasyfikatora na wynikach tych modeli. Model meta-klasyfikatora przetwarza wyniki z kilku modeli i uzyskuje ogólny wynik. Ta metoda umożliwia wykorzystanie różnych rodzajów modeli i pozwala na lepsze dopasowanie do złożonych zbiorów danych.

Ensemble learning jest popularną techniką w uczeniu maszynowym i znajduje zastosowanie w wielu dziedzinach, takich jak rozpoznawanie obrazów, rozpoznawanie mowy, klasyfikacja tekstu i wiele innych.
