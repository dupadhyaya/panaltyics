# -*- coding: utf-8 -*-
"""
Sun Mar 11 11:06:57 2018: Dhiraj  : Orange
"""
import Orange
from Orange.widgets.visualize.owruleviewer import OWRuleViewer
from AnyQt.QtWidgets import QApplication
from Orange.classification import CN2Learner
data = Orange.data.Table("Titanic")
learner = Orange.classification.CN2Learner()
model = learner(data)
model.instance = data
a = QApplication([])
ow = OWRuleViewer()
ow.set_classifier(model) # not working
ow.show()
a.exec()
