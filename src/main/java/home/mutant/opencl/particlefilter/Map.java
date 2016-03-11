package home.mutant.opencl.particlefilter;

import java.util.List;

import home.mutant.dl.models.Image;
import home.mutant.dl.models.ImageFloat;

public class Map {
	Image map;
	public Map(List<Image> images, int dimNoImages){
		int dimImages = (int) Math.sqrt(images.get(0).getDataFloat().length);
		int i=0;
		map = new ImageFloat(dimImages*dimNoImages, dimImages*dimNoImages);
		for(int y=0;y<dimNoImages;y++){
			for(int x=0;x<dimNoImages;x++){
				map.pasteImage(images.get(i++), x*dimImages, y*dimImages);
			}
		}
	}
}
