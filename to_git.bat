@echo off
set now=%date% %time%
del *.log
git add -A
git commit -m "Обновление: %now%"
git push
