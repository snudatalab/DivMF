cd src;
unzip ../data/gowalla-15.zip -d ../data/gowalla-15;
python main.py --data gowalla-15 --reclen 13 --unmask True --ut 500 --rbs 5000 --cbs 5000
