import csv
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from ohdeer.features import integral,rect_sum,make_features,extract
from ohdeer.geometry import iou,nms,negative_is_clean,windows
from ohdeer.data import convert_csv,convert_coco,validate_manifest,write_json,split_manifest
from ohdeer.pipeline import detection_metrics


class FeaturesTest(unittest.TestCase):
    def test_summed_area_and_haar_against_pixels(self):
        gray=np.random.default_rng(10).random((47,61))
        ii=integral(gray);iis=integral(gray**2)
        boxes=np.array([[0,0,16,16],[3,7,60,45],[0,0,61,47]])
        np.testing.assert_allclose(rect_sum(ii,boxes),[gray[y1:y2,x1:x2].sum() for x1,y1,x2,y2 in boxes])
        spec=make_features(30)
        actual=extract(ii,iis,boxes,spec)
        expected=np.zeros_like(actual)
        for j,(x,y,r,b) in enumerate(boxes):
            patch=gray[y:b,x:r];std=max(patch.std(),1/255)
            for k,rects in enumerate(spec['templates']):
                for x1,y1,x2,y2,weight in rects:
                    expected[j,k]+=weight*patch[(b-y)*y1//8:(b-y)*y2//8,
                                                (r-x)*x1//8:(r-x)*x2//8].mean()/std
        np.testing.assert_allclose(actual,expected,atol=2e-6)

    def test_constant_image_has_no_contrast(self):
        image=np.full((32,40),.7)
        result=extract(integral(image),integral(image**2),[[0,0,40,32]],make_features(30))
        np.testing.assert_allclose(result,0,atol=1e-8)


class GeometryTest(unittest.TestCase):
    def test_iou_nms_and_partial_deer_negatives(self):
        np.testing.assert_allclose(iou([0,0,10,10],[[0,0,10,10],[10,0,20,10]]),[1,0])
        np.testing.assert_allclose(iou(['0','0','10','10'],[[0,0,10,10]]),[1])
        np.testing.assert_array_equal(nms([[0,0,10,10],[1,1,10,10],[20,20,30,30]],[.9,.8,.7]),[0,2])
        self.assertFalse(negative_is_clean(np.array([4,4,6,6]),[[0,0,10,10]]))
        self.assertTrue(negative_is_clean(np.array([20,20,30,30]),[[0,0,10,10]]))

    def test_scan_covers_border(self):
        candidates=list(windows(100,70,[.4],[1],.3))
        self.assertIn((60,30,100,70),candidates)


class MetricsTest(unittest.TestCase):
    def prediction(self,id,box,score):
        return dict(zip(['image_id','xmin','ymin','xmax','ymax','score'],[id,*box,score]))

    def test_duplicate_is_false_positive_and_missing_images_count(self):
        records=[{'image_id':'a','boxes':[[0,0,10,10]]},{'image_id':'b','boxes':[[0,0,10,10]]},
                 {'image_id':'empty','boxes':[]}]
        predictions=[self.prediction('a',[0,0,10,10],3),self.prediction('a',[0,0,10,10],2),
                     self.prediction('empty',[0,0,10,10],1)]
        m=detection_metrics(records,predictions)
        self.assertEqual((m['true_positive'],m['false_positive'],m['false_negative']),(1,2,1))
        self.assertAlmostEqual(m['recall'],.5);self.assertAlmostEqual(m['precision'],1/3)
        self.assertAlmostEqual(m['ap_at_iou'],.5)

    def test_empty_predictions_and_negative_threshold(self):
        records=[{'image_id':'a','boxes':[[0,0,10,10]]}]
        self.assertEqual(detection_metrics(records,[])['ap_at_iou'],0)
        predictions=[self.prediction('a',[0,0,10,10],-.5)]
        m=detection_metrics(records,predictions,score_threshold=0)
        self.assertEqual(m['true_positive'],0);self.assertEqual(m['ap_at_iou'],1)


class DatasetTest(unittest.TestCase):
    def test_conversion_and_group_split(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for i in range(10):
                Image.fromarray(np.random.default_rng(i).integers(0,255,(30,40),dtype='uint8')).save(root/f'{i}.png')
            source=root/'raw.csv'
            with source.open('w',newline='') as f:
                writer=csv.writer(f);writer.writerow(['file','x','y','w','h','group'])
                for i in range(10):writer.writerow([f'{i}.png',.25,.2,.5,.6,str(i//2)])
            data=convert_csv(source,root,'file',['x','y','w','h'],'xywh',True,'group',True)
            self.assertEqual(data['images'][0]['boxes'],[[10,6,30,24]])
            write_json(root/'manifest.json',data);data=validate_manifest(root/'manifest.json',root)
            data=split_manifest(data,root/'split.csv')
            by_group={}
            for r in data['images']:
                by_group.setdefault(r['group_id'],set()).add(r['split'])
            self.assertTrue(all(len(v)==1 for v in by_group.values()))

    def test_reject_cross_split_duplicates(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);shared=root/'shared.csv'
            shared.write_text('image_id,split\na,train\nb,val\n')
            data={'images':[{'image_id':'a','sha256':'same'},{'image_id':'b','sha256':'same'}]}
            with self.assertRaises(ValueError):split_manifest(data,root/'s.csv',shared=shared)

    def test_reject_invalid_boxes_and_implicit_split(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            data={'images':[{'image_id':'a','sha256':'a'},{'image_id':'b','sha256':'b'}]}
            with self.assertRaises(ValueError):split_manifest(data,root/'s.csv')

    def test_coco_requires_targets_and_rejects_crowd(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'coco.json'
            write_json(path,{'images':[{'id':1,'file_name':'a.jpg'}],
                 'annotations':[{'image_id':1,'category_id':3,'bbox':[1,2,3,4]}]})
            result=convert_coco(path,{3},True)
            self.assertEqual(result['images'][0]['boxes'],[[1,2,4,6]])
            data={'images':[{'id':1,'file_name':'a.jpg'}],
                 'annotations':[{'image_id':1,'category_id':3,'bbox':[1,2,3,4],'iscrowd':1}]}
            write_json(path,data)
            with self.assertRaises(ValueError):convert_coco(path,{3},True)


class MatchingRegressionTest(unittest.TestCase):
    def test_matches_second_available_overlapping_truth(self):
        records=[{'image_id':'x','boxes':[[0,0,10,10],[2,0,12,10]]}]
        predictions=[dict(image_id='x',xmin=0,ymin=0,xmax=10,ymax=10,score=score)
                     for score in [2,1]]
        metrics=detection_metrics(records,predictions)
        self.assertEqual(metrics['true_positive'],2)
        self.assertEqual(metrics['false_negative'],0)


if __name__ == "__main__":
    unittest.main()
