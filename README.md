# icdl2016language

Source code accompanying the paper

Michael Spranger and Katrien Beuls (2016) [Referential Uncertainty and Word Learning in High-dimensional, Continuous Meaning Spaces](https://www.dropbox.com/s/6dbwkjdg2u1e64n/referential-uncertainty-word.pdf). In Development and Learning and Epigenetic Robotics (ICDL-Epirob), 2016 Joint IEEE International Conferences on, 2016. IEEE.

Requires python >=2.7, bokeh, numpy, scikit-learn, keras (for MLP)

For the grounded experiments you need https://github.com/mspranger/data-qrio-objects next to this repository.

To get started please check general_multiclass_multilabel.py. Hyper-parameter grid optimization is in in general_multiclass_multilabel_optimize.py. The MLP training and testing is in general_multiclass_multilabel_mlp3.py. Functions for description games are in description_game.py. robot_data.py handles data loading and pre-processing.

LICENSE
-------
Copyright (c) 2016, Michael Spranger (http://www.michael-spranger.com).
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

  * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above
    copyright notice, this list of conditions and the following
    disclaimer in the documentation and/or other materials
    provided with the distribution.
    
THIS SOFTWARE IS PROVIDED BY THE AUTHOR 'AS IS' AND ANY EXPRESSED
OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
