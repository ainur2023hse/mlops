## Описание:
В данном репозитории представлен код для тренировки и получения инференса модели классификации изображений. Изображения классифицируются на основе того, изображена ли на них собака

## Тренировка модели:
```python commands.py train```
## Инференс:
```python commands.py infer```

Результатом выполнения команды является ```.csv``` фаил, который содержит лейблы (1 - есть собака, 0 - нету собаки) и
вероятности, полученные из модели.