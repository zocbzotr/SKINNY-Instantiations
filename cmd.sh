#! /bin/bash

cd ./ZOCB
echo "Timing ZOCB..."
for i in {1..2}
do
./ZOCB_Timing
done
cd -
cd ./ZOTR
echo "Timing ZOTR..."
for i in {1..2}
do
./ZOTR_Timing
done
cd -
cd ./ThetaCB3
echo "Timing ThetaCB3..."
for i in {1..2}
do
./ThetaCB3_Timing
done
cd -
