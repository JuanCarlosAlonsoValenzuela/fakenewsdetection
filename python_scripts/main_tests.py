'''
    En este archivo se ejecutan todas las pruebas del proyecto, para que se ejecute correctamente deben cumplirse las siguientes condiciones:
        - El archivo subset_test.csv debe encontrarse en el mismo directorio que este archivo
        - El archivo tests.py debe encontrarse en el mismo directorio que este archivo
        - El archivo ticnn_preprocessed.csv debe encontrarse en la carpeta datasets del proyecto
        - El archivo model_creation.py debe encontrarse en el mismo directorio que este archivo
        - El archivo preprocessing.py debe encontrarse en el mismo directorio que este archivo
        - El archivo model_training.py debe encontrarse en el mismo directorio que este archivo
        - El archivo hparams_optimization.py debe encontrarse en el mismo directorio que este archivo
'''

import tests

print('################ PRUEBAS SPRINT 1 ################')
tests.prueba1()
tests.prueba2()
tests.prueba3()
tests.prueba4()
print('################ PRUEBAS SPRINT 1 FINALIZADAS ################\n\n')

print('################ PRUEBAS SPRINT 2 ################')
tests.prueba5()
tests.prueba6()
tests.prueba7()
tests.prueba8()
tests.prueba9()
print('################ PRUEBAS SPRINT 2 FINALIZADAS ################\n\n')

print('################ PRUEBAS SPRINT 3 ################')
tests.prueba10()
print('################ PRUEBAS SPRINT 3 FINALIZADAS ################\n\n')

print('################ PRUEBAS SPRINT 4 ################')
# Esta prueba es la más exigente computacionalmente (tomó 40 minutos ejecutarla en el equipo 1, de 16GB de RAM)
# Por este motivo se ha dejado comentada, para evitar posibles problemas de memoria en función del equipo en el que
# se ejecute. Si se elimina el comentario, puede ejecutarse sin problema alguno

# tests.prueba11()
print('################ PRUEBAS SPRINT 4 FINALIZADAS ################\n\n')




