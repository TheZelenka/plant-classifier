import os
import torch
import streamlit as st
from PIL import Image
from torchvision import transforms, models

#НАСТРОЙКИ
DATA_DIR = "C:\\Users\\sanya\\plant_leaf_classifier\\data"
MODEL_PATH = "C:\\Users\\sanya\\PyCharmMiscProject\\best_model.pth"


# 1. Получаем список классов без загрузки всего датасета
@st.cache_data
def get_class_names(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    # PyTorch ImageFolder сортирует классы по алфавиту, сделаем так же:
    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    return classes


# 2. Загружаем модель в кэш Streamlit
@st.cache_resource
def load_model(model_path, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.shufflenet_v2_x1_0(weights=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device


# 3. Функция предсказания (теперь принимает объект PIL Image)
def predict_image(img, model, class_names, device, img_size=224):
    tfm = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Переводим в RGB, если вдруг загрузят черно-белое фото или PNG с прозрачностью
    img = img.convert('RGB')
    x = tfm(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)
        probs = torch.nn.functional.softmax(out, dim=1)
        top_prob, top_idx = torch.topk(probs, k=1)

    top_prob = top_prob.cpu().numpy()[0]
    top_idx = top_idx.cpu().numpy()[0]

    results = [(class_names[i], float(top_prob[j])) for j, i in enumerate(top_idx)]
    return results


#ИНТЕРФЕЙС STREAMLIT

st.title(" Определение вида растения по листу")
st.write("Загрузите фотографию листа, и нейросеть определит его вид.")

# Подготавливаем классы и модель
try:
    class_names = get_class_names(DATA_DIR)
    model, device = load_model(MODEL_PATH, num_classes=len(class_names))
except Exception as e:
    st.error(f"Ошибка при загрузке модели или данных: {e}")
    st.stop()  # Останавливаем работу, если пути указаны неверно

# Виджет загрузки
uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Открываем изображение
    image = Image.open(uploaded_file)

    # Показываем картинку
    st.image(image, caption="Загруженное изображение", use_container_width=True)
    st.write("Идёт распознавание...")

    # Делаем предсказание
    try:
        results = predict_image(image, model, class_names, device)
        best_class, best_prob = results[0]

        # Выводим красивый результат
        st.success(f"**Результат:** Это {best_class}")
        st.info(f"**Уверенность нейросети:** {best_prob * 100:.2f}%")

    except Exception as e:
        st.error(f"Произошла ошибка при обработке изображения: {e}")





