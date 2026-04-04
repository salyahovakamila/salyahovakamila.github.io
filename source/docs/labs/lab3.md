#  Лабораторная работа №3
## Тема: CI/CD для статического сайта в SourceCraft

##  Цель работы: Реализовать сценарий автоматического развёртывания статического сайта, построенного на движке MkDocs, с использованием платформы SourceCraft.


##  Задание
- Реализовать сценарий автоматического развёртывания статического сайта на MkDocs с использованием SourceCraft
- Реализовать сценарий автоматического развёртывания этого же сайта с помощью GitHub Actions
- В рамках одного локального репозитория добавить 2 удалённых репозитория (SourceCraft и GitHub)
- Продемонстрировать результаты выполнения

##  Выполнение
# Шаг 1: Регистрация и создание организации

1) Авторизовалась на https://sourcecraft.dev через Яндекс-аккаунт
2) Создала публичную организацию: salyahova-k
3) Создала пустой публичный репозиторий: portfolio
# Шаг 2: Генерация персонального токена (PAT)

1) Перешла в Настройки → Personal Access Tokens
2) Создала новый токен с параметрами
3) Скопировала токен для использования в Git

```bash
# Добавление удалённого репозитория SourceCraft
git remote add sourcecraft https://salyahova-k:<TOKEN>@git.sourcecraft.dev/salyahova-k/portfolio.git

# Проверка
git remote -v
# Результат:
# origin        https://github.com/salyahovakamila/salyahovakamila.github.io.git
# sourcecraft   https://git.sourcecraft.dev/salyahova-k/portfolio.git
```
# requirements.txt
```text
mkdocs>=1.5.0
mkdocs-material>=9.0.0
pymdown-extensions>=10.0
pygments>=2.0
markdown>=3.3
```
# Отправка кода в репозитории
```bash
# Добавление файлов
git add .
git commit -m "Initial commit: MkDocs site with CI/CD"

# Отправка в оба репозитория
git push origin main        # GitHub
git push sourcecraft main   # SourceCraft
```
# Проблемы и их решение
# ERROR: You must give at least one requirement to install
- Создан файл requirements.txt с зависимостями
- Исправлена команда в CI/CD: pip install → pip install -r requirements.txt
##  Выводы
### Работа с несколькими remote-репозиториями в одном локальном проекте
- Создание и использование Personal Access Tokens (PAT)
- Настройка CI/CD пайплайнов в SourceCraft
- Создание GitHub Actions workflows
- Конфигурация MkDocs для статических сайтов
- Автоматический деплой статических сайтов