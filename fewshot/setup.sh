mkdir ../data/omniglot
cd ../data/omniglot

# Download omniglot dataset -- https://github.com/brendenlake/omniglot
mkdir ../omniglot_resized
git clone https://github.com/brendenlake/omniglot.git
mv omniglot/python/images_background.zip ../omniglot_resized
mv omniglot/python/images_evaluation.zip ../omniglot_resized
cd ../omniglot_resized
unzip images_background.zip
unzip images_evaluation.zip
cd ../

# Preprocess images -- see https://github.com/cbfinn/maml 
rm -rf omniglot
cd omniglot_resized
cp ../../fewshot/omniglot_resize_images.py ./
mv images_background/* ./
mv images_evaluation/* ./
rm -r images_background/
rm -r images_evaluation/
rm images_background.zip
rm images_evaluation.zip
python omniglot_resize_images.py
rm omniglot_resize_images.py
cd ../
rm -rf omniglot

