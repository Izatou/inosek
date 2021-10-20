sudo rm -r ~/.config/.enose-model
mkdir ~/.config/.enose-model
wget https://inose.id:5555/api/update/downloadModel?serial_number=1
mv 'downloadModel?serial_number=1' ~/.config/.enose-model/model.zip
cd ~/.config/.enose-model/
unzip model.zip
wget https://raw.githubusercontent.com/Izatou/inosek/main/index.py
sudo cp index.py ~/.config/.enose-model/

echo Beress gan
