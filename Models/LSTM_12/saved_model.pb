¬ø!
Å
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
°
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements(
handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintÿÿÿÿÿÿÿÿÿ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68 
|
salt_pred/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*!
shared_namesalt_pred/kernel
u
$salt_pred/kernel/Read/ReadVariableOpReadVariableOpsalt_pred/kernel*
_output_shapes

:@*
dtype0
t
salt_pred/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namesalt_pred/bias
m
"salt_pred/bias/Read/ReadVariableOpReadVariableOpsalt_pred/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0

salt_seq/lstm_cell_54/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*-
shared_namesalt_seq/lstm_cell_54/kernel

0salt_seq/lstm_cell_54/kernel/Read/ReadVariableOpReadVariableOpsalt_seq/lstm_cell_54/kernel*
_output_shapes
:	*
dtype0
©
&salt_seq/lstm_cell_54/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *7
shared_name(&salt_seq/lstm_cell_54/recurrent_kernel
¢
:salt_seq/lstm_cell_54/recurrent_kernel/Read/ReadVariableOpReadVariableOp&salt_seq/lstm_cell_54/recurrent_kernel*
_output_shapes
:	 *
dtype0

salt_seq/lstm_cell_54/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namesalt_seq/lstm_cell_54/bias

.salt_seq/lstm_cell_54/bias/Read/ReadVariableOpReadVariableOpsalt_seq/lstm_cell_54/bias*
_output_shapes	
:*
dtype0

qty_seq/lstm_cell_55/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*,
shared_nameqty_seq/lstm_cell_55/kernel

/qty_seq/lstm_cell_55/kernel/Read/ReadVariableOpReadVariableOpqty_seq/lstm_cell_55/kernel*
_output_shapes
:	*
dtype0
§
%qty_seq/lstm_cell_55/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *6
shared_name'%qty_seq/lstm_cell_55/recurrent_kernel
 
9qty_seq/lstm_cell_55/recurrent_kernel/Read/ReadVariableOpReadVariableOp%qty_seq/lstm_cell_55/recurrent_kernel*
_output_shapes
:	 *
dtype0

qty_seq/lstm_cell_55/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameqty_seq/lstm_cell_55/bias

-qty_seq/lstm_cell_55/bias/Read/ReadVariableOpReadVariableOpqty_seq/lstm_cell_55/bias*
_output_shapes	
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

Adam/salt_pred/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*(
shared_nameAdam/salt_pred/kernel/m

+Adam/salt_pred/kernel/m/Read/ReadVariableOpReadVariableOpAdam/salt_pred/kernel/m*
_output_shapes

:@*
dtype0

Adam/salt_pred/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/salt_pred/bias/m
{
)Adam/salt_pred/bias/m/Read/ReadVariableOpReadVariableOpAdam/salt_pred/bias/m*
_output_shapes
:*
dtype0
£
#Adam/salt_seq/lstm_cell_54/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*4
shared_name%#Adam/salt_seq/lstm_cell_54/kernel/m

7Adam/salt_seq/lstm_cell_54/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/salt_seq/lstm_cell_54/kernel/m*
_output_shapes
:	*
dtype0
·
-Adam/salt_seq/lstm_cell_54/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *>
shared_name/-Adam/salt_seq/lstm_cell_54/recurrent_kernel/m
°
AAdam/salt_seq/lstm_cell_54/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp-Adam/salt_seq/lstm_cell_54/recurrent_kernel/m*
_output_shapes
:	 *
dtype0

!Adam/salt_seq/lstm_cell_54/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/salt_seq/lstm_cell_54/bias/m

5Adam/salt_seq/lstm_cell_54/bias/m/Read/ReadVariableOpReadVariableOp!Adam/salt_seq/lstm_cell_54/bias/m*
_output_shapes	
:*
dtype0
¡
"Adam/qty_seq/lstm_cell_55/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*3
shared_name$"Adam/qty_seq/lstm_cell_55/kernel/m

6Adam/qty_seq/lstm_cell_55/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/qty_seq/lstm_cell_55/kernel/m*
_output_shapes
:	*
dtype0
µ
,Adam/qty_seq/lstm_cell_55/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *=
shared_name.,Adam/qty_seq/lstm_cell_55/recurrent_kernel/m
®
@Adam/qty_seq/lstm_cell_55/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/qty_seq/lstm_cell_55/recurrent_kernel/m*
_output_shapes
:	 *
dtype0

 Adam/qty_seq/lstm_cell_55/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/qty_seq/lstm_cell_55/bias/m

4Adam/qty_seq/lstm_cell_55/bias/m/Read/ReadVariableOpReadVariableOp Adam/qty_seq/lstm_cell_55/bias/m*
_output_shapes	
:*
dtype0

Adam/salt_pred/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*(
shared_nameAdam/salt_pred/kernel/v

+Adam/salt_pred/kernel/v/Read/ReadVariableOpReadVariableOpAdam/salt_pred/kernel/v*
_output_shapes

:@*
dtype0

Adam/salt_pred/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/salt_pred/bias/v
{
)Adam/salt_pred/bias/v/Read/ReadVariableOpReadVariableOpAdam/salt_pred/bias/v*
_output_shapes
:*
dtype0
£
#Adam/salt_seq/lstm_cell_54/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*4
shared_name%#Adam/salt_seq/lstm_cell_54/kernel/v

7Adam/salt_seq/lstm_cell_54/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/salt_seq/lstm_cell_54/kernel/v*
_output_shapes
:	*
dtype0
·
-Adam/salt_seq/lstm_cell_54/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *>
shared_name/-Adam/salt_seq/lstm_cell_54/recurrent_kernel/v
°
AAdam/salt_seq/lstm_cell_54/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp-Adam/salt_seq/lstm_cell_54/recurrent_kernel/v*
_output_shapes
:	 *
dtype0

!Adam/salt_seq/lstm_cell_54/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/salt_seq/lstm_cell_54/bias/v

5Adam/salt_seq/lstm_cell_54/bias/v/Read/ReadVariableOpReadVariableOp!Adam/salt_seq/lstm_cell_54/bias/v*
_output_shapes	
:*
dtype0
¡
"Adam/qty_seq/lstm_cell_55/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*3
shared_name$"Adam/qty_seq/lstm_cell_55/kernel/v

6Adam/qty_seq/lstm_cell_55/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/qty_seq/lstm_cell_55/kernel/v*
_output_shapes
:	*
dtype0
µ
,Adam/qty_seq/lstm_cell_55/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *=
shared_name.,Adam/qty_seq/lstm_cell_55/recurrent_kernel/v
®
@Adam/qty_seq/lstm_cell_55/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/qty_seq/lstm_cell_55/recurrent_kernel/v*
_output_shapes
:	 *
dtype0

 Adam/qty_seq/lstm_cell_55/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/qty_seq/lstm_cell_55/bias/v

4Adam/qty_seq/lstm_cell_55/bias/v/Read/ReadVariableOpReadVariableOp Adam/qty_seq/lstm_cell_55/bias/v*
_output_shapes	
:*
dtype0

NoOpNoOp
ùG
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*´G
valueªGB§G B G

layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
* 
Á
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
*&call_and_return_all_conditional_losses*
Á
cell

state_spec
	variables
trainable_variables
regularization_losses
 	keras_api
!_random_generator
"__call__
*#&call_and_return_all_conditional_losses*
¥
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(_random_generator
)__call__
**&call_and_return_all_conditional_losses* 
¥
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/_random_generator
0__call__
*1&call_and_return_all_conditional_losses* 

2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses* 
¦

8kernel
9bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses*
ä
@iter

Abeta_1

Bbeta_2
	Cdecay
Dlearning_rate8m9mEmFmGmHmImJm8v9vEvFvGvHvIvJv*
<
E0
F1
G2
H3
I4
J5
86
97*
<
E0
F1
G2
H3
I4
J5
86
97*
* 
°
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

Pserving_default* 
ã
Q
state_size

Ekernel
Frecurrent_kernel
Gbias
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V_random_generator
W__call__
*X&call_and_return_all_conditional_losses*
* 

E0
F1
G2*

E0
F1
G2*
* 


Ystates
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
ã
_
state_size

Hkernel
Irecurrent_kernel
Jbias
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d_random_generator
e__call__
*f&call_and_return_all_conditional_losses*
* 

H0
I1
J2*

H0
I1
J2*
* 


gstates
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
	variables
trainable_variables
regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 

mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
$	variables
%trainable_variables
&regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 

rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
+	variables
,trainable_variables
-regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 

wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEsalt_pred/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEsalt_pred/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

80
91*

80
91*
* 

|non_trainable_variables

}layers
~metrics
layer_regularization_losses
layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEsalt_seq/lstm_cell_54/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&salt_seq/lstm_cell_54/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEsalt_seq/lstm_cell_54/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEqty_seq/lstm_cell_55/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%qty_seq/lstm_cell_55/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEqty_seq/lstm_cell_55/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
* 
<
0
1
2
3
4
5
6
7*

0*
* 
* 
* 
* 

E0
F1
G2*

E0
F1
G2*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
R	variables
Strainable_variables
Tregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

0*
* 
* 
* 
* 

H0
I1
J2*

H0
I1
J2*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
`	variables
atrainable_variables
bregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<

total

count
	variables
	keras_api*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
}
VARIABLE_VALUEAdam/salt_pred/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/salt_pred/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/salt_seq/lstm_cell_54/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE-Adam/salt_seq/lstm_cell_54/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE!Adam/salt_seq/lstm_cell_54/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/qty_seq/lstm_cell_55/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/qty_seq/lstm_cell_55/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/qty_seq/lstm_cell_55/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/salt_pred/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/salt_pred/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/salt_seq/lstm_cell_54/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE-Adam/salt_seq/lstm_cell_54/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE!Adam/salt_seq/lstm_cell_54/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/qty_seq/lstm_cell_55/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/qty_seq/lstm_cell_55/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/qty_seq/lstm_cell_55/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_quantity_dataPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ

serving_default_salt_dataPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ
Ç
StatefulPartitionedCallStatefulPartitionedCallserving_default_quantity_dataserving_default_salt_dataqty_seq/lstm_cell_55/kernel%qty_seq/lstm_cell_55/recurrent_kernelqty_seq/lstm_cell_55/biassalt_seq/lstm_cell_54/kernel&salt_seq/lstm_cell_54/recurrent_kernelsalt_seq/lstm_cell_54/biassalt_pred/kernelsalt_pred/bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_865463
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$salt_pred/kernel/Read/ReadVariableOp"salt_pred/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp0salt_seq/lstm_cell_54/kernel/Read/ReadVariableOp:salt_seq/lstm_cell_54/recurrent_kernel/Read/ReadVariableOp.salt_seq/lstm_cell_54/bias/Read/ReadVariableOp/qty_seq/lstm_cell_55/kernel/Read/ReadVariableOp9qty_seq/lstm_cell_55/recurrent_kernel/Read/ReadVariableOp-qty_seq/lstm_cell_55/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/salt_pred/kernel/m/Read/ReadVariableOp)Adam/salt_pred/bias/m/Read/ReadVariableOp7Adam/salt_seq/lstm_cell_54/kernel/m/Read/ReadVariableOpAAdam/salt_seq/lstm_cell_54/recurrent_kernel/m/Read/ReadVariableOp5Adam/salt_seq/lstm_cell_54/bias/m/Read/ReadVariableOp6Adam/qty_seq/lstm_cell_55/kernel/m/Read/ReadVariableOp@Adam/qty_seq/lstm_cell_55/recurrent_kernel/m/Read/ReadVariableOp4Adam/qty_seq/lstm_cell_55/bias/m/Read/ReadVariableOp+Adam/salt_pred/kernel/v/Read/ReadVariableOp)Adam/salt_pred/bias/v/Read/ReadVariableOp7Adam/salt_seq/lstm_cell_54/kernel/v/Read/ReadVariableOpAAdam/salt_seq/lstm_cell_54/recurrent_kernel/v/Read/ReadVariableOp5Adam/salt_seq/lstm_cell_54/bias/v/Read/ReadVariableOp6Adam/qty_seq/lstm_cell_55/kernel/v/Read/ReadVariableOp@Adam/qty_seq/lstm_cell_55/recurrent_kernel/v/Read/ReadVariableOp4Adam/qty_seq/lstm_cell_55/bias/v/Read/ReadVariableOpConst*,
Tin%
#2!	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_save_867094
¡	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamesalt_pred/kernelsalt_pred/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratesalt_seq/lstm_cell_54/kernel&salt_seq/lstm_cell_54/recurrent_kernelsalt_seq/lstm_cell_54/biasqty_seq/lstm_cell_55/kernel%qty_seq/lstm_cell_55/recurrent_kernelqty_seq/lstm_cell_55/biastotalcountAdam/salt_pred/kernel/mAdam/salt_pred/bias/m#Adam/salt_seq/lstm_cell_54/kernel/m-Adam/salt_seq/lstm_cell_54/recurrent_kernel/m!Adam/salt_seq/lstm_cell_54/bias/m"Adam/qty_seq/lstm_cell_55/kernel/m,Adam/qty_seq/lstm_cell_55/recurrent_kernel/m Adam/qty_seq/lstm_cell_55/bias/mAdam/salt_pred/kernel/vAdam/salt_pred/bias/v#Adam/salt_seq/lstm_cell_54/kernel/v-Adam/salt_seq/lstm_cell_54/recurrent_kernel/v!Adam/salt_seq/lstm_cell_54/bias/v"Adam/qty_seq/lstm_cell_55/kernel/v,Adam/qty_seq/lstm_cell_55/recurrent_kernel/v Adam/qty_seq/lstm_cell_55/bias/v*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_restore_867197×æ
¨J

C__inference_qty_seq_layer_call_and_return_conditional_losses_864633

inputs>
+lstm_cell_55_matmul_readvariableop_resource:	@
-lstm_cell_55_matmul_1_readvariableop_resource:	 ;
,lstm_cell_55_biasadd_readvariableop_resource:	
identity¢#lstm_cell_55/BiasAdd/ReadVariableOp¢"lstm_cell_55/MatMul/ReadVariableOp¢$lstm_cell_55/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_55/MatMul/ReadVariableOpReadVariableOp+lstm_cell_55_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_55/MatMulMatMulstrided_slice_2:output:0*lstm_cell_55/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_55/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_55_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0
lstm_cell_55/MatMul_1MatMulzeros:output:0,lstm_cell_55/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_55/addAddV2lstm_cell_55/MatMul:product:0lstm_cell_55/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_55/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_55_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_55/BiasAddBiasAddlstm_cell_55/add:z:0+lstm_cell_55/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_55/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstm_cell_55/splitSplit%lstm_cell_55/split/split_dim:output:0lstm_cell_55/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitn
lstm_cell_55/SigmoidSigmoidlstm_cell_55/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
lstm_cell_55/Sigmoid_1Sigmoidlstm_cell_55/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
lstm_cell_55/mulMullstm_cell_55/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ h
lstm_cell_55/ReluRelulstm_cell_55/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_55/mul_1Mullstm_cell_55/Sigmoid:y:0lstm_cell_55/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ {
lstm_cell_55/add_1AddV2lstm_cell_55/mul:z:0lstm_cell_55/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
lstm_cell_55/Sigmoid_2Sigmoidlstm_cell_55/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ e
lstm_cell_55/Relu_1Relulstm_cell_55/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_55/mul_2Mullstm_cell_55/Sigmoid_2:y:0!lstm_cell_55/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_55_matmul_readvariableop_resource-lstm_cell_55_matmul_1_readvariableop_resource,lstm_cell_55_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_864549*
condR
while_cond_864548*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ À
NoOpNoOp$^lstm_cell_55/BiasAdd/ReadVariableOp#^lstm_cell_55/MatMul/ReadVariableOp%^lstm_cell_55/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_55/BiasAdd/ReadVariableOp#lstm_cell_55/BiasAdd/ReadVariableOp2H
"lstm_cell_55/MatMul/ReadVariableOp"lstm_cell_55/MatMul/ReadVariableOp2L
$lstm_cell_55/MatMul_1/ReadVariableOp$lstm_cell_55/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
Ã
while_cond_865565
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_865565___redundant_placeholder04
0while_while_cond_865565___redundant_placeholder14
0while_while_cond_865565___redundant_placeholder24
0while_while_cond_865565___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:


e
F__inference_salt_seq_2_layer_call_and_return_conditional_losses_866722

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seed[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
8
Ð
while_body_865995
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_54_matmul_readvariableop_resource_0:	H
5while_lstm_cell_54_matmul_1_readvariableop_resource_0:	 C
4while_lstm_cell_54_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_54_matmul_readvariableop_resource:	F
3while_lstm_cell_54_matmul_1_readvariableop_resource:	 A
2while_lstm_cell_54_biasadd_readvariableop_resource:	¢)while/lstm_cell_54/BiasAdd/ReadVariableOp¢(while/lstm_cell_54/MatMul/ReadVariableOp¢*while/lstm_cell_54/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_54/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_54_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0º
while/lstm_cell_54/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_54/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
*while/lstm_cell_54/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_54_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0¡
while/lstm_cell_54/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_54/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_54/addAddV2#while/lstm_cell_54/MatMul:product:0%while/lstm_cell_54/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_54/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_54_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstm_cell_54/BiasAddBiasAddwhile/lstm_cell_54/add:z:01while/lstm_cell_54/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_54/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
while/lstm_cell_54/splitSplit+while/lstm_cell_54/split/split_dim:output:0#while/lstm_cell_54/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitz
while/lstm_cell_54/SigmoidSigmoid!while/lstm_cell_54/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
while/lstm_cell_54/Sigmoid_1Sigmoid!while/lstm_cell_54/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_54/mulMul while/lstm_cell_54/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
while/lstm_cell_54/ReluRelu!while/lstm_cell_54/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_54/mul_1Mulwhile/lstm_cell_54/Sigmoid:y:0%while/lstm_cell_54/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_54/add_1AddV2while/lstm_cell_54/mul:z:0while/lstm_cell_54/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
while/lstm_cell_54/Sigmoid_2Sigmoid!while/lstm_cell_54/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q
while/lstm_cell_54/Relu_1Reluwhile/lstm_cell_54/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_54/mul_2Mul while/lstm_cell_54/Sigmoid_2:y:0'while/lstm_cell_54/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Å
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_54/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒy
while/Identity_4Identitywhile/lstm_cell_54/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ y
while/Identity_5Identitywhile/lstm_cell_54/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ð

while/NoOpNoOp*^while/lstm_cell_54/BiasAdd/ReadVariableOp)^while/lstm_cell_54/MatMul/ReadVariableOp+^while/lstm_cell_54/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_54_biasadd_readvariableop_resource4while_lstm_cell_54_biasadd_readvariableop_resource_0"l
3while_lstm_cell_54_matmul_1_readvariableop_resource5while_lstm_cell_54_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_54_matmul_readvariableop_resource3while_lstm_cell_54_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2V
)while/lstm_cell_54/BiasAdd/ReadVariableOp)while/lstm_cell_54/BiasAdd/ReadVariableOp2T
(while/lstm_cell_54/MatMul/ReadVariableOp(while/lstm_cell_54/MatMul/ReadVariableOp2X
*while/lstm_cell_54/MatMul_1/ReadVariableOp*while/lstm_cell_54/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 

¸
)__inference_salt_seq_layer_call_fn_865485
inputs_0
unknown:	
	unknown_0:	 
	unknown_1:	
identity¢StatefulPartitionedCallè
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_salt_seq_layer_call_and_return_conditional_losses_863522o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
ºM

#model_12_salt_seq_while_body_863087@
<model_12_salt_seq_while_model_12_salt_seq_while_loop_counterF
Bmodel_12_salt_seq_while_model_12_salt_seq_while_maximum_iterations'
#model_12_salt_seq_while_placeholder)
%model_12_salt_seq_while_placeholder_1)
%model_12_salt_seq_while_placeholder_2)
%model_12_salt_seq_while_placeholder_3?
;model_12_salt_seq_while_model_12_salt_seq_strided_slice_1_0{
wmodel_12_salt_seq_while_tensorarrayv2read_tensorlistgetitem_model_12_salt_seq_tensorarrayunstack_tensorlistfromtensor_0X
Emodel_12_salt_seq_while_lstm_cell_54_matmul_readvariableop_resource_0:	Z
Gmodel_12_salt_seq_while_lstm_cell_54_matmul_1_readvariableop_resource_0:	 U
Fmodel_12_salt_seq_while_lstm_cell_54_biasadd_readvariableop_resource_0:	$
 model_12_salt_seq_while_identity&
"model_12_salt_seq_while_identity_1&
"model_12_salt_seq_while_identity_2&
"model_12_salt_seq_while_identity_3&
"model_12_salt_seq_while_identity_4&
"model_12_salt_seq_while_identity_5=
9model_12_salt_seq_while_model_12_salt_seq_strided_slice_1y
umodel_12_salt_seq_while_tensorarrayv2read_tensorlistgetitem_model_12_salt_seq_tensorarrayunstack_tensorlistfromtensorV
Cmodel_12_salt_seq_while_lstm_cell_54_matmul_readvariableop_resource:	X
Emodel_12_salt_seq_while_lstm_cell_54_matmul_1_readvariableop_resource:	 S
Dmodel_12_salt_seq_while_lstm_cell_54_biasadd_readvariableop_resource:	¢;model_12/salt_seq/while/lstm_cell_54/BiasAdd/ReadVariableOp¢:model_12/salt_seq/while/lstm_cell_54/MatMul/ReadVariableOp¢<model_12/salt_seq/while/lstm_cell_54/MatMul_1/ReadVariableOp
Imodel_12/salt_seq/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
;model_12/salt_seq/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemwmodel_12_salt_seq_while_tensorarrayv2read_tensorlistgetitem_model_12_salt_seq_tensorarrayunstack_tensorlistfromtensor_0#model_12_salt_seq_while_placeholderRmodel_12/salt_seq/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0Á
:model_12/salt_seq/while/lstm_cell_54/MatMul/ReadVariableOpReadVariableOpEmodel_12_salt_seq_while_lstm_cell_54_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0ð
+model_12/salt_seq/while/lstm_cell_54/MatMulMatMulBmodel_12/salt_seq/while/TensorArrayV2Read/TensorListGetItem:item:0Bmodel_12/salt_seq/while/lstm_cell_54/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
<model_12/salt_seq/while/lstm_cell_54/MatMul_1/ReadVariableOpReadVariableOpGmodel_12_salt_seq_while_lstm_cell_54_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0×
-model_12/salt_seq/while/lstm_cell_54/MatMul_1MatMul%model_12_salt_seq_while_placeholder_2Dmodel_12/salt_seq/while/lstm_cell_54/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ
(model_12/salt_seq/while/lstm_cell_54/addAddV25model_12/salt_seq/while/lstm_cell_54/MatMul:product:07model_12/salt_seq/while/lstm_cell_54/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
;model_12/salt_seq/while/lstm_cell_54/BiasAdd/ReadVariableOpReadVariableOpFmodel_12_salt_seq_while_lstm_cell_54_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0Ý
,model_12/salt_seq/while/lstm_cell_54/BiasAddBiasAdd,model_12/salt_seq/while/lstm_cell_54/add:z:0Cmodel_12/salt_seq/while/lstm_cell_54/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
4model_12/salt_seq/while/lstm_cell_54/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¥
*model_12/salt_seq/while/lstm_cell_54/splitSplit=model_12/salt_seq/while/lstm_cell_54/split/split_dim:output:05model_12/salt_seq/while/lstm_cell_54/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split
,model_12/salt_seq/while/lstm_cell_54/SigmoidSigmoid3model_12/salt_seq/while/lstm_cell_54/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
.model_12/salt_seq/while/lstm_cell_54/Sigmoid_1Sigmoid3model_12/salt_seq/while/lstm_cell_54/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¼
(model_12/salt_seq/while/lstm_cell_54/mulMul2model_12/salt_seq/while/lstm_cell_54/Sigmoid_1:y:0%model_12_salt_seq_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
)model_12/salt_seq/while/lstm_cell_54/ReluRelu3model_12/salt_seq/while/lstm_cell_54/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Î
*model_12/salt_seq/while/lstm_cell_54/mul_1Mul0model_12/salt_seq/while/lstm_cell_54/Sigmoid:y:07model_12/salt_seq/while/lstm_cell_54/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ã
*model_12/salt_seq/while/lstm_cell_54/add_1AddV2,model_12/salt_seq/while/lstm_cell_54/mul:z:0.model_12/salt_seq/while/lstm_cell_54/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
.model_12/salt_seq/while/lstm_cell_54/Sigmoid_2Sigmoid3model_12/salt_seq/while/lstm_cell_54/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
+model_12/salt_seq/while/lstm_cell_54/Relu_1Relu.model_12/salt_seq/while/lstm_cell_54/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ò
*model_12/salt_seq/while/lstm_cell_54/mul_2Mul2model_12/salt_seq/while/lstm_cell_54/Sigmoid_2:y:09model_12/salt_seq/while/lstm_cell_54/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
<model_12/salt_seq/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem%model_12_salt_seq_while_placeholder_1#model_12_salt_seq_while_placeholder.model_12/salt_seq/while/lstm_cell_54/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒ_
model_12/salt_seq/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
model_12/salt_seq/while/addAddV2#model_12_salt_seq_while_placeholder&model_12/salt_seq/while/add/y:output:0*
T0*
_output_shapes
: a
model_12/salt_seq/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :¯
model_12/salt_seq/while/add_1AddV2<model_12_salt_seq_while_model_12_salt_seq_while_loop_counter(model_12/salt_seq/while/add_1/y:output:0*
T0*
_output_shapes
: 
 model_12/salt_seq/while/IdentityIdentity!model_12/salt_seq/while/add_1:z:0^model_12/salt_seq/while/NoOp*
T0*
_output_shapes
: ²
"model_12/salt_seq/while/Identity_1IdentityBmodel_12_salt_seq_while_model_12_salt_seq_while_maximum_iterations^model_12/salt_seq/while/NoOp*
T0*
_output_shapes
: 
"model_12/salt_seq/while/Identity_2Identitymodel_12/salt_seq/while/add:z:0^model_12/salt_seq/while/NoOp*
T0*
_output_shapes
: Ï
"model_12/salt_seq/while/Identity_3IdentityLmodel_12/salt_seq/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^model_12/salt_seq/while/NoOp*
T0*
_output_shapes
: :éèÒ¯
"model_12/salt_seq/while/Identity_4Identity.model_12/salt_seq/while/lstm_cell_54/mul_2:z:0^model_12/salt_seq/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¯
"model_12/salt_seq/while/Identity_5Identity.model_12/salt_seq/while/lstm_cell_54/add_1:z:0^model_12/salt_seq/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
model_12/salt_seq/while/NoOpNoOp<^model_12/salt_seq/while/lstm_cell_54/BiasAdd/ReadVariableOp;^model_12/salt_seq/while/lstm_cell_54/MatMul/ReadVariableOp=^model_12/salt_seq/while/lstm_cell_54/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "M
 model_12_salt_seq_while_identity)model_12/salt_seq/while/Identity:output:0"Q
"model_12_salt_seq_while_identity_1+model_12/salt_seq/while/Identity_1:output:0"Q
"model_12_salt_seq_while_identity_2+model_12/salt_seq/while/Identity_2:output:0"Q
"model_12_salt_seq_while_identity_3+model_12/salt_seq/while/Identity_3:output:0"Q
"model_12_salt_seq_while_identity_4+model_12/salt_seq/while/Identity_4:output:0"Q
"model_12_salt_seq_while_identity_5+model_12/salt_seq/while/Identity_5:output:0"
Dmodel_12_salt_seq_while_lstm_cell_54_biasadd_readvariableop_resourceFmodel_12_salt_seq_while_lstm_cell_54_biasadd_readvariableop_resource_0"
Emodel_12_salt_seq_while_lstm_cell_54_matmul_1_readvariableop_resourceGmodel_12_salt_seq_while_lstm_cell_54_matmul_1_readvariableop_resource_0"
Cmodel_12_salt_seq_while_lstm_cell_54_matmul_readvariableop_resourceEmodel_12_salt_seq_while_lstm_cell_54_matmul_readvariableop_resource_0"x
9model_12_salt_seq_while_model_12_salt_seq_strided_slice_1;model_12_salt_seq_while_model_12_salt_seq_strided_slice_1_0"ð
umodel_12_salt_seq_while_tensorarrayv2read_tensorlistgetitem_model_12_salt_seq_tensorarrayunstack_tensorlistfromtensorwmodel_12_salt_seq_while_tensorarrayv2read_tensorlistgetitem_model_12_salt_seq_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2z
;model_12/salt_seq/while/lstm_cell_54/BiasAdd/ReadVariableOp;model_12/salt_seq/while/lstm_cell_54/BiasAdd/ReadVariableOp2x
:model_12/salt_seq/while/lstm_cell_54/MatMul/ReadVariableOp:model_12/salt_seq/while/lstm_cell_54/MatMul/ReadVariableOp2|
<model_12/salt_seq/while/lstm_cell_54/MatMul_1/ReadVariableOp<model_12/salt_seq/while/lstm_cell_54/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 

´
D__inference_model_12_layer_call_and_return_conditional_losses_864694

inputs
inputs_1!
qty_seq_864671:	!
qty_seq_864673:	 
qty_seq_864675:	"
salt_seq_864678:	"
salt_seq_864680:	 
salt_seq_864682:	"
salt_pred_864688:@
salt_pred_864690:
identity¢qty_seq/StatefulPartitionedCall¢!qty_seq_2/StatefulPartitionedCall¢!salt_pred/StatefulPartitionedCall¢ salt_seq/StatefulPartitionedCall¢"salt_seq_2/StatefulPartitionedCall
qty_seq/StatefulPartitionedCallStatefulPartitionedCallinputs_1qty_seq_864671qty_seq_864673qty_seq_864675*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_qty_seq_layer_call_and_return_conditional_losses_864633
 salt_seq/StatefulPartitionedCallStatefulPartitionedCallinputssalt_seq_864678salt_seq_864680salt_seq_864682*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_salt_seq_layer_call_and_return_conditional_losses_864468ï
"salt_seq_2/StatefulPartitionedCallStatefulPartitionedCall)salt_seq/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_salt_seq_2_layer_call_and_return_conditional_losses_864309
!qty_seq_2/StatefulPartitionedCallStatefulPartitionedCall(qty_seq/StatefulPartitionedCall:output:0#^salt_seq_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_qty_seq_2_layer_call_and_return_conditional_losses_864286
pattern/PartitionedCallPartitionedCall+salt_seq_2/StatefulPartitionedCall:output:0*qty_seq_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_pattern_layer_call_and_return_conditional_losses_864211
!salt_pred/StatefulPartitionedCallStatefulPartitionedCall pattern/PartitionedCall:output:0salt_pred_864688salt_pred_864690*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_salt_pred_layer_call_and_return_conditional_losses_864223y
IdentityIdentity*salt_pred/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿø
NoOpNoOp ^qty_seq/StatefulPartitionedCall"^qty_seq_2/StatefulPartitionedCall"^salt_pred/StatefulPartitionedCall!^salt_seq/StatefulPartitionedCall#^salt_seq_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2B
qty_seq/StatefulPartitionedCallqty_seq/StatefulPartitionedCall2F
!qty_seq_2/StatefulPartitionedCall!qty_seq_2/StatefulPartitionedCall2F
!salt_pred/StatefulPartitionedCall!salt_pred/StatefulPartitionedCall2D
 salt_seq/StatefulPartitionedCall salt_seq/StatefulPartitionedCall2H
"salt_seq_2/StatefulPartitionedCall"salt_seq_2/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:SO
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
©J

D__inference_salt_seq_layer_call_and_return_conditional_losses_864468

inputs>
+lstm_cell_54_matmul_readvariableop_resource:	@
-lstm_cell_54_matmul_1_readvariableop_resource:	 ;
,lstm_cell_54_biasadd_readvariableop_resource:	
identity¢#lstm_cell_54/BiasAdd/ReadVariableOp¢"lstm_cell_54/MatMul/ReadVariableOp¢$lstm_cell_54/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_54/MatMul/ReadVariableOpReadVariableOp+lstm_cell_54_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_54/MatMulMatMulstrided_slice_2:output:0*lstm_cell_54/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_54/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_54_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0
lstm_cell_54/MatMul_1MatMulzeros:output:0,lstm_cell_54/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_54/addAddV2lstm_cell_54/MatMul:product:0lstm_cell_54/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_54/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_54_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_54/BiasAddBiasAddlstm_cell_54/add:z:0+lstm_cell_54/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_54/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstm_cell_54/splitSplit%lstm_cell_54/split/split_dim:output:0lstm_cell_54/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitn
lstm_cell_54/SigmoidSigmoidlstm_cell_54/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
lstm_cell_54/Sigmoid_1Sigmoidlstm_cell_54/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
lstm_cell_54/mulMullstm_cell_54/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ h
lstm_cell_54/ReluRelulstm_cell_54/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_54/mul_1Mullstm_cell_54/Sigmoid:y:0lstm_cell_54/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ {
lstm_cell_54/add_1AddV2lstm_cell_54/mul:z:0lstm_cell_54/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
lstm_cell_54/Sigmoid_2Sigmoidlstm_cell_54/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ e
lstm_cell_54/Relu_1Relulstm_cell_54/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_54/mul_2Mullstm_cell_54/Sigmoid_2:y:0!lstm_cell_54/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_54_matmul_readvariableop_resource-lstm_cell_54_matmul_1_readvariableop_resource,lstm_cell_54_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_864384*
condR
while_cond_864383*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ À
NoOpNoOp$^lstm_cell_54/BiasAdd/ReadVariableOp#^lstm_cell_54/MatMul/ReadVariableOp%^lstm_cell_54/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_54/BiasAdd/ReadVariableOp#lstm_cell_54/BiasAdd/ReadVariableOp2H
"lstm_cell_54/MatMul/ReadVariableOp"lstm_cell_54/MatMul/ReadVariableOp2L
$lstm_cell_54/MatMul_1/ReadVariableOp$lstm_cell_54/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


"model_12_qty_seq_while_cond_862947>
:model_12_qty_seq_while_model_12_qty_seq_while_loop_counterD
@model_12_qty_seq_while_model_12_qty_seq_while_maximum_iterations&
"model_12_qty_seq_while_placeholder(
$model_12_qty_seq_while_placeholder_1(
$model_12_qty_seq_while_placeholder_2(
$model_12_qty_seq_while_placeholder_3@
<model_12_qty_seq_while_less_model_12_qty_seq_strided_slice_1V
Rmodel_12_qty_seq_while_model_12_qty_seq_while_cond_862947___redundant_placeholder0V
Rmodel_12_qty_seq_while_model_12_qty_seq_while_cond_862947___redundant_placeholder1V
Rmodel_12_qty_seq_while_model_12_qty_seq_while_cond_862947___redundant_placeholder2V
Rmodel_12_qty_seq_while_model_12_qty_seq_while_cond_862947___redundant_placeholder3#
model_12_qty_seq_while_identity
¦
model_12/qty_seq/while/LessLess"model_12_qty_seq_while_placeholder<model_12_qty_seq_while_less_model_12_qty_seq_strided_slice_1*
T0*
_output_shapes
: m
model_12/qty_seq/while/IdentityIdentitymodel_12/qty_seq/while/Less:z:0*
T0
*
_output_shapes
: "K
model_12_qty_seq_while_identity(model_12/qty_seq/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
8
Ð
while_body_864098
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_54_matmul_readvariableop_resource_0:	H
5while_lstm_cell_54_matmul_1_readvariableop_resource_0:	 C
4while_lstm_cell_54_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_54_matmul_readvariableop_resource:	F
3while_lstm_cell_54_matmul_1_readvariableop_resource:	 A
2while_lstm_cell_54_biasadd_readvariableop_resource:	¢)while/lstm_cell_54/BiasAdd/ReadVariableOp¢(while/lstm_cell_54/MatMul/ReadVariableOp¢*while/lstm_cell_54/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_54/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_54_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0º
while/lstm_cell_54/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_54/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
*while/lstm_cell_54/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_54_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0¡
while/lstm_cell_54/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_54/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_54/addAddV2#while/lstm_cell_54/MatMul:product:0%while/lstm_cell_54/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_54/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_54_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstm_cell_54/BiasAddBiasAddwhile/lstm_cell_54/add:z:01while/lstm_cell_54/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_54/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
while/lstm_cell_54/splitSplit+while/lstm_cell_54/split/split_dim:output:0#while/lstm_cell_54/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitz
while/lstm_cell_54/SigmoidSigmoid!while/lstm_cell_54/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
while/lstm_cell_54/Sigmoid_1Sigmoid!while/lstm_cell_54/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_54/mulMul while/lstm_cell_54/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
while/lstm_cell_54/ReluRelu!while/lstm_cell_54/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_54/mul_1Mulwhile/lstm_cell_54/Sigmoid:y:0%while/lstm_cell_54/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_54/add_1AddV2while/lstm_cell_54/mul:z:0while/lstm_cell_54/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
while/lstm_cell_54/Sigmoid_2Sigmoid!while/lstm_cell_54/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q
while/lstm_cell_54/Relu_1Reluwhile/lstm_cell_54/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_54/mul_2Mul while/lstm_cell_54/Sigmoid_2:y:0'while/lstm_cell_54/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Å
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_54/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒy
while/Identity_4Identitywhile/lstm_cell_54/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ y
while/Identity_5Identitywhile/lstm_cell_54/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ð

while/NoOpNoOp*^while/lstm_cell_54/BiasAdd/ReadVariableOp)^while/lstm_cell_54/MatMul/ReadVariableOp+^while/lstm_cell_54/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_54_biasadd_readvariableop_resource4while_lstm_cell_54_biasadd_readvariableop_resource_0"l
3while_lstm_cell_54_matmul_1_readvariableop_resource5while_lstm_cell_54_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_54_matmul_readvariableop_resource3while_lstm_cell_54_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2V
)while/lstm_cell_54/BiasAdd/ReadVariableOp)while/lstm_cell_54/BiasAdd/ReadVariableOp2T
(while/lstm_cell_54/MatMul/ReadVariableOp(while/lstm_cell_54/MatMul/ReadVariableOp2X
*while/lstm_cell_54/MatMul_1/ReadVariableOp*while/lstm_cell_54/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
ù
¶
)__inference_salt_seq_layer_call_fn_865507

inputs
unknown:	
	unknown_0:	 
	unknown_1:	
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_salt_seq_layer_call_and_return_conditional_losses_864468o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
îB
ð

salt_seq_while_body_865038.
*salt_seq_while_salt_seq_while_loop_counter4
0salt_seq_while_salt_seq_while_maximum_iterations
salt_seq_while_placeholder 
salt_seq_while_placeholder_1 
salt_seq_while_placeholder_2 
salt_seq_while_placeholder_3-
)salt_seq_while_salt_seq_strided_slice_1_0i
esalt_seq_while_tensorarrayv2read_tensorlistgetitem_salt_seq_tensorarrayunstack_tensorlistfromtensor_0O
<salt_seq_while_lstm_cell_54_matmul_readvariableop_resource_0:	Q
>salt_seq_while_lstm_cell_54_matmul_1_readvariableop_resource_0:	 L
=salt_seq_while_lstm_cell_54_biasadd_readvariableop_resource_0:	
salt_seq_while_identity
salt_seq_while_identity_1
salt_seq_while_identity_2
salt_seq_while_identity_3
salt_seq_while_identity_4
salt_seq_while_identity_5+
'salt_seq_while_salt_seq_strided_slice_1g
csalt_seq_while_tensorarrayv2read_tensorlistgetitem_salt_seq_tensorarrayunstack_tensorlistfromtensorM
:salt_seq_while_lstm_cell_54_matmul_readvariableop_resource:	O
<salt_seq_while_lstm_cell_54_matmul_1_readvariableop_resource:	 J
;salt_seq_while_lstm_cell_54_biasadd_readvariableop_resource:	¢2salt_seq/while/lstm_cell_54/BiasAdd/ReadVariableOp¢1salt_seq/while/lstm_cell_54/MatMul/ReadVariableOp¢3salt_seq/while/lstm_cell_54/MatMul_1/ReadVariableOp
@salt_seq/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ó
2salt_seq/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemesalt_seq_while_tensorarrayv2read_tensorlistgetitem_salt_seq_tensorarrayunstack_tensorlistfromtensor_0salt_seq_while_placeholderIsalt_seq/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0¯
1salt_seq/while/lstm_cell_54/MatMul/ReadVariableOpReadVariableOp<salt_seq_while_lstm_cell_54_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0Õ
"salt_seq/while/lstm_cell_54/MatMulMatMul9salt_seq/while/TensorArrayV2Read/TensorListGetItem:item:09salt_seq/while/lstm_cell_54/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
3salt_seq/while/lstm_cell_54/MatMul_1/ReadVariableOpReadVariableOp>salt_seq_while_lstm_cell_54_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0¼
$salt_seq/while/lstm_cell_54/MatMul_1MatMulsalt_seq_while_placeholder_2;salt_seq/while/lstm_cell_54/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
salt_seq/while/lstm_cell_54/addAddV2,salt_seq/while/lstm_cell_54/MatMul:product:0.salt_seq/while/lstm_cell_54/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
2salt_seq/while/lstm_cell_54/BiasAdd/ReadVariableOpReadVariableOp=salt_seq_while_lstm_cell_54_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0Â
#salt_seq/while/lstm_cell_54/BiasAddBiasAdd#salt_seq/while/lstm_cell_54/add:z:0:salt_seq/while/lstm_cell_54/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
+salt_seq/while/lstm_cell_54/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
!salt_seq/while/lstm_cell_54/splitSplit4salt_seq/while/lstm_cell_54/split/split_dim:output:0,salt_seq/while/lstm_cell_54/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split
#salt_seq/while/lstm_cell_54/SigmoidSigmoid*salt_seq/while/lstm_cell_54/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
%salt_seq/while/lstm_cell_54/Sigmoid_1Sigmoid*salt_seq/while/lstm_cell_54/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¡
salt_seq/while/lstm_cell_54/mulMul)salt_seq/while/lstm_cell_54/Sigmoid_1:y:0salt_seq_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 salt_seq/while/lstm_cell_54/ReluRelu*salt_seq/while/lstm_cell_54/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ³
!salt_seq/while/lstm_cell_54/mul_1Mul'salt_seq/while/lstm_cell_54/Sigmoid:y:0.salt_seq/while/lstm_cell_54/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¨
!salt_seq/while/lstm_cell_54/add_1AddV2#salt_seq/while/lstm_cell_54/mul:z:0%salt_seq/while/lstm_cell_54/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
%salt_seq/while/lstm_cell_54/Sigmoid_2Sigmoid*salt_seq/while/lstm_cell_54/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"salt_seq/while/lstm_cell_54/Relu_1Relu%salt_seq/while/lstm_cell_54/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ·
!salt_seq/while/lstm_cell_54/mul_2Mul)salt_seq/while/lstm_cell_54/Sigmoid_2:y:00salt_seq/while/lstm_cell_54/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ é
3salt_seq/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemsalt_seq_while_placeholder_1salt_seq_while_placeholder%salt_seq/while/lstm_cell_54/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒV
salt_seq/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :w
salt_seq/while/addAddV2salt_seq_while_placeholdersalt_seq/while/add/y:output:0*
T0*
_output_shapes
: X
salt_seq/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
salt_seq/while/add_1AddV2*salt_seq_while_salt_seq_while_loop_countersalt_seq/while/add_1/y:output:0*
T0*
_output_shapes
: t
salt_seq/while/IdentityIdentitysalt_seq/while/add_1:z:0^salt_seq/while/NoOp*
T0*
_output_shapes
: 
salt_seq/while/Identity_1Identity0salt_seq_while_salt_seq_while_maximum_iterations^salt_seq/while/NoOp*
T0*
_output_shapes
: t
salt_seq/while/Identity_2Identitysalt_seq/while/add:z:0^salt_seq/while/NoOp*
T0*
_output_shapes
: ´
salt_seq/while/Identity_3IdentityCsalt_seq/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^salt_seq/while/NoOp*
T0*
_output_shapes
: :éèÒ
salt_seq/while/Identity_4Identity%salt_seq/while/lstm_cell_54/mul_2:z:0^salt_seq/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
salt_seq/while/Identity_5Identity%salt_seq/while/lstm_cell_54/add_1:z:0^salt_seq/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ô
salt_seq/while/NoOpNoOp3^salt_seq/while/lstm_cell_54/BiasAdd/ReadVariableOp2^salt_seq/while/lstm_cell_54/MatMul/ReadVariableOp4^salt_seq/while/lstm_cell_54/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ";
salt_seq_while_identity salt_seq/while/Identity:output:0"?
salt_seq_while_identity_1"salt_seq/while/Identity_1:output:0"?
salt_seq_while_identity_2"salt_seq/while/Identity_2:output:0"?
salt_seq_while_identity_3"salt_seq/while/Identity_3:output:0"?
salt_seq_while_identity_4"salt_seq/while/Identity_4:output:0"?
salt_seq_while_identity_5"salt_seq/while/Identity_5:output:0"|
;salt_seq_while_lstm_cell_54_biasadd_readvariableop_resource=salt_seq_while_lstm_cell_54_biasadd_readvariableop_resource_0"~
<salt_seq_while_lstm_cell_54_matmul_1_readvariableop_resource>salt_seq_while_lstm_cell_54_matmul_1_readvariableop_resource_0"z
:salt_seq_while_lstm_cell_54_matmul_readvariableop_resource<salt_seq_while_lstm_cell_54_matmul_readvariableop_resource_0"T
'salt_seq_while_salt_seq_strided_slice_1)salt_seq_while_salt_seq_strided_slice_1_0"Ì
csalt_seq_while_tensorarrayv2read_tensorlistgetitem_salt_seq_tensorarrayunstack_tensorlistfromtensoresalt_seq_while_tensorarrayv2read_tensorlistgetitem_salt_seq_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2h
2salt_seq/while/lstm_cell_54/BiasAdd/ReadVariableOp2salt_seq/while/lstm_cell_54/BiasAdd/ReadVariableOp2f
1salt_seq/while/lstm_cell_54/MatMul/ReadVariableOp1salt_seq/while/lstm_cell_54/MatMul/ReadVariableOp2j
3salt_seq/while/lstm_cell_54/MatMul_1/ReadVariableOp3salt_seq/while/lstm_cell_54/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
8

C__inference_qty_seq_layer_call_and_return_conditional_losses_863681

inputs&
lstm_cell_55_863599:	&
lstm_cell_55_863601:	 "
lstm_cell_55_863603:	
identity¢$lstm_cell_55/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskõ
$lstm_cell_55/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_55_863599lstm_cell_55_863601lstm_cell_55_863603*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_55_layer_call_and_return_conditional_losses_863598n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ·
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_55_863599lstm_cell_55_863601lstm_cell_55_863603*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_863612*
condR
while_cond_863611*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ u
NoOpNoOp%^lstm_cell_55/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2L
$lstm_cell_55/StatefulPartitionedCall$lstm_cell_55/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
Ã
while_cond_864097
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_864097___redundant_placeholder04
0while_while_cond_864097___redundant_placeholder14
0while_while_cond_864097___redundant_placeholder24
0while_while_cond_864097___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
Ù
d
F__inference_salt_seq_2_layer_call_and_return_conditional_losses_864195

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
²

÷
salt_seq_while_cond_865037.
*salt_seq_while_salt_seq_while_loop_counter4
0salt_seq_while_salt_seq_while_maximum_iterations
salt_seq_while_placeholder 
salt_seq_while_placeholder_1 
salt_seq_while_placeholder_2 
salt_seq_while_placeholder_30
,salt_seq_while_less_salt_seq_strided_slice_1F
Bsalt_seq_while_salt_seq_while_cond_865037___redundant_placeholder0F
Bsalt_seq_while_salt_seq_while_cond_865037___redundant_placeholder1F
Bsalt_seq_while_salt_seq_while_cond_865037___redundant_placeholder2F
Bsalt_seq_while_salt_seq_while_cond_865037___redundant_placeholder3
salt_seq_while_identity

salt_seq/while/LessLesssalt_seq_while_placeholder,salt_seq_while_less_salt_seq_strided_slice_1*
T0*
_output_shapes
: ]
salt_seq/while/IdentityIdentitysalt_seq/while/Less:z:0*
T0
*
_output_shapes
: ";
salt_seq_while_identity salt_seq/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
µ
Ã
while_cond_863611
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_863611___redundant_placeholder04
0while_while_cond_863611___redundant_placeholder14
0while_while_cond_863611___redundant_placeholder24
0while_while_cond_863611___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
µ
Ã
while_cond_863802
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_863802___redundant_placeholder04
0while_while_cond_863802___redundant_placeholder14
0while_while_cond_863802___redundant_placeholder24
0while_while_cond_863802___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
8
Ð
while_body_866325
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_55_matmul_readvariableop_resource_0:	H
5while_lstm_cell_55_matmul_1_readvariableop_resource_0:	 C
4while_lstm_cell_55_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_55_matmul_readvariableop_resource:	F
3while_lstm_cell_55_matmul_1_readvariableop_resource:	 A
2while_lstm_cell_55_biasadd_readvariableop_resource:	¢)while/lstm_cell_55/BiasAdd/ReadVariableOp¢(while/lstm_cell_55/MatMul/ReadVariableOp¢*while/lstm_cell_55/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_55/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_55_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0º
while/lstm_cell_55/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_55/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
*while/lstm_cell_55/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_55_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0¡
while/lstm_cell_55/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_55/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_55/addAddV2#while/lstm_cell_55/MatMul:product:0%while/lstm_cell_55/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_55/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_55_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstm_cell_55/BiasAddBiasAddwhile/lstm_cell_55/add:z:01while/lstm_cell_55/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_55/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
while/lstm_cell_55/splitSplit+while/lstm_cell_55/split/split_dim:output:0#while/lstm_cell_55/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitz
while/lstm_cell_55/SigmoidSigmoid!while/lstm_cell_55/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
while/lstm_cell_55/Sigmoid_1Sigmoid!while/lstm_cell_55/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_55/mulMul while/lstm_cell_55/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
while/lstm_cell_55/ReluRelu!while/lstm_cell_55/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_55/mul_1Mulwhile/lstm_cell_55/Sigmoid:y:0%while/lstm_cell_55/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_55/add_1AddV2while/lstm_cell_55/mul:z:0while/lstm_cell_55/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
while/lstm_cell_55/Sigmoid_2Sigmoid!while/lstm_cell_55/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q
while/lstm_cell_55/Relu_1Reluwhile/lstm_cell_55/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_55/mul_2Mul while/lstm_cell_55/Sigmoid_2:y:0'while/lstm_cell_55/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Å
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_55/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒy
while/Identity_4Identitywhile/lstm_cell_55/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ y
while/Identity_5Identitywhile/lstm_cell_55/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ð

while/NoOpNoOp*^while/lstm_cell_55/BiasAdd/ReadVariableOp)^while/lstm_cell_55/MatMul/ReadVariableOp+^while/lstm_cell_55/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_55_biasadd_readvariableop_resource4while_lstm_cell_55_biasadd_readvariableop_resource_0"l
3while_lstm_cell_55_matmul_1_readvariableop_resource5while_lstm_cell_55_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_55_matmul_readvariableop_resource3while_lstm_cell_55_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2V
)while/lstm_cell_55/BiasAdd/ReadVariableOp)while/lstm_cell_55/BiasAdd/ReadVariableOp2T
(while/lstm_cell_55/MatMul/ReadVariableOp(while/lstm_cell_55/MatMul/ReadVariableOp2X
*while/lstm_cell_55/MatMul_1/ReadVariableOp*while/lstm_cell_55/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 


ã
qty_seq_while_cond_864898,
(qty_seq_while_qty_seq_while_loop_counter2
.qty_seq_while_qty_seq_while_maximum_iterations
qty_seq_while_placeholder
qty_seq_while_placeholder_1
qty_seq_while_placeholder_2
qty_seq_while_placeholder_3.
*qty_seq_while_less_qty_seq_strided_slice_1D
@qty_seq_while_qty_seq_while_cond_864898___redundant_placeholder0D
@qty_seq_while_qty_seq_while_cond_864898___redundant_placeholder1D
@qty_seq_while_qty_seq_while_cond_864898___redundant_placeholder2D
@qty_seq_while_qty_seq_while_cond_864898___redundant_placeholder3
qty_seq_while_identity

qty_seq/while/LessLessqty_seq_while_placeholder*qty_seq_while_less_qty_seq_strided_slice_1*
T0*
_output_shapes
: [
qty_seq/while/IdentityIdentityqty_seq/while/Less:z:0*
T0
*
_output_shapes
: "9
qty_seq_while_identityqty_seq/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
ï

Ø
)__inference_model_12_layer_call_fn_864735
	salt_data
quantity_data
unknown:	
	unknown_0:	 
	unknown_1:	
	unknown_2:	
	unknown_3:	 
	unknown_4:	
	unknown_5:@
	unknown_6:
identity¢StatefulPartitionedCallº
StatefulPartitionedCallStatefulPartitionedCall	salt_dataquantity_dataunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_model_12_layer_call_and_return_conditional_losses_864694o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	salt_data:ZV
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namequantity_data
ÿ	
d
E__inference_qty_seq_2_layer_call_and_return_conditional_losses_864286

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seed[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
8
Ð
while_body_864384
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_54_matmul_readvariableop_resource_0:	H
5while_lstm_cell_54_matmul_1_readvariableop_resource_0:	 C
4while_lstm_cell_54_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_54_matmul_readvariableop_resource:	F
3while_lstm_cell_54_matmul_1_readvariableop_resource:	 A
2while_lstm_cell_54_biasadd_readvariableop_resource:	¢)while/lstm_cell_54/BiasAdd/ReadVariableOp¢(while/lstm_cell_54/MatMul/ReadVariableOp¢*while/lstm_cell_54/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_54/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_54_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0º
while/lstm_cell_54/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_54/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
*while/lstm_cell_54/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_54_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0¡
while/lstm_cell_54/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_54/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_54/addAddV2#while/lstm_cell_54/MatMul:product:0%while/lstm_cell_54/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_54/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_54_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstm_cell_54/BiasAddBiasAddwhile/lstm_cell_54/add:z:01while/lstm_cell_54/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_54/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
while/lstm_cell_54/splitSplit+while/lstm_cell_54/split/split_dim:output:0#while/lstm_cell_54/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitz
while/lstm_cell_54/SigmoidSigmoid!while/lstm_cell_54/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
while/lstm_cell_54/Sigmoid_1Sigmoid!while/lstm_cell_54/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_54/mulMul while/lstm_cell_54/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
while/lstm_cell_54/ReluRelu!while/lstm_cell_54/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_54/mul_1Mulwhile/lstm_cell_54/Sigmoid:y:0%while/lstm_cell_54/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_54/add_1AddV2while/lstm_cell_54/mul:z:0while/lstm_cell_54/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
while/lstm_cell_54/Sigmoid_2Sigmoid!while/lstm_cell_54/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q
while/lstm_cell_54/Relu_1Reluwhile/lstm_cell_54/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_54/mul_2Mul while/lstm_cell_54/Sigmoid_2:y:0'while/lstm_cell_54/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Å
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_54/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒy
while/Identity_4Identitywhile/lstm_cell_54/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ y
while/Identity_5Identitywhile/lstm_cell_54/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ð

while/NoOpNoOp*^while/lstm_cell_54/BiasAdd/ReadVariableOp)^while/lstm_cell_54/MatMul/ReadVariableOp+^while/lstm_cell_54/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_54_biasadd_readvariableop_resource4while_lstm_cell_54_biasadd_readvariableop_resource_0"l
3while_lstm_cell_54_matmul_1_readvariableop_resource5while_lstm_cell_54_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_54_matmul_readvariableop_resource3while_lstm_cell_54_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2V
)while/lstm_cell_54/BiasAdd/ReadVariableOp)while/lstm_cell_54/BiasAdd/ReadVariableOp2T
(while/lstm_cell_54/MatMul/ReadVariableOp(while/lstm_cell_54/MatMul/ReadVariableOp2X
*while/lstm_cell_54/MatMul_1/ReadVariableOp*while/lstm_cell_54/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
Ý

H__inference_lstm_cell_54_layer_call_and_return_conditional_losses_866847

inputs
states_0
states_11
matmul_readvariableop_resource:	3
 matmul_1_readvariableop_resource:	 .
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ N
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/1
µ
Ã
while_cond_866467
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_866467___redundant_placeholder04
0while_while_cond_866467___redundant_placeholder14
0while_while_cond_866467___redundant_placeholder24
0while_while_cond_866467___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
ë
ö
-__inference_lstm_cell_54_layer_call_fn_866798

inputs
states_0
states_1
unknown:	
	unknown_0:	 
	unknown_1:	
identity

identity_1

identity_2¢StatefulPartitionedCall¨
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_54_layer_call_and_return_conditional_losses_863248o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/1


ã
qty_seq_while_cond_865191,
(qty_seq_while_qty_seq_while_loop_counter2
.qty_seq_while_qty_seq_while_maximum_iterations
qty_seq_while_placeholder
qty_seq_while_placeholder_1
qty_seq_while_placeholder_2
qty_seq_while_placeholder_3.
*qty_seq_while_less_qty_seq_strided_slice_1D
@qty_seq_while_qty_seq_while_cond_865191___redundant_placeholder0D
@qty_seq_while_qty_seq_while_cond_865191___redundant_placeholder1D
@qty_seq_while_qty_seq_while_cond_865191___redundant_placeholder2D
@qty_seq_while_qty_seq_while_cond_865191___redundant_placeholder3
qty_seq_while_identity

qty_seq/while/LessLessqty_seq_while_placeholder*qty_seq_while_less_qty_seq_strided_slice_1*
T0*
_output_shapes
: [
qty_seq/while/IdentityIdentityqty_seq/while/Less:z:0*
T0
*
_output_shapes
: "9
qty_seq_while_identityqty_seq/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
Ä

*__inference_salt_pred_layer_call_fn_866771

inputs
unknown:@
	unknown_0:
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_salt_pred_layer_call_and_return_conditional_losses_864223o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
µ
Ã
while_cond_863261
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_863261___redundant_placeholder04
0while_while_cond_863261___redundant_placeholder14
0while_while_cond_863261___redundant_placeholder24
0while_while_cond_863261___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
µ
Ã
while_cond_863452
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_863452___redundant_placeholder04
0while_while_cond_863452___redundant_placeholder14
0while_while_cond_863452___redundant_placeholder24
0while_while_cond_863452___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
Ý

H__inference_lstm_cell_55_layer_call_and_return_conditional_losses_866945

inputs
states_0
states_11
matmul_readvariableop_resource:	3
 matmul_1_readvariableop_resource:	 .
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ N
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/1
ØA
Ð

qty_seq_while_body_865192,
(qty_seq_while_qty_seq_while_loop_counter2
.qty_seq_while_qty_seq_while_maximum_iterations
qty_seq_while_placeholder
qty_seq_while_placeholder_1
qty_seq_while_placeholder_2
qty_seq_while_placeholder_3+
'qty_seq_while_qty_seq_strided_slice_1_0g
cqty_seq_while_tensorarrayv2read_tensorlistgetitem_qty_seq_tensorarrayunstack_tensorlistfromtensor_0N
;qty_seq_while_lstm_cell_55_matmul_readvariableop_resource_0:	P
=qty_seq_while_lstm_cell_55_matmul_1_readvariableop_resource_0:	 K
<qty_seq_while_lstm_cell_55_biasadd_readvariableop_resource_0:	
qty_seq_while_identity
qty_seq_while_identity_1
qty_seq_while_identity_2
qty_seq_while_identity_3
qty_seq_while_identity_4
qty_seq_while_identity_5)
%qty_seq_while_qty_seq_strided_slice_1e
aqty_seq_while_tensorarrayv2read_tensorlistgetitem_qty_seq_tensorarrayunstack_tensorlistfromtensorL
9qty_seq_while_lstm_cell_55_matmul_readvariableop_resource:	N
;qty_seq_while_lstm_cell_55_matmul_1_readvariableop_resource:	 I
:qty_seq_while_lstm_cell_55_biasadd_readvariableop_resource:	¢1qty_seq/while/lstm_cell_55/BiasAdd/ReadVariableOp¢0qty_seq/while/lstm_cell_55/MatMul/ReadVariableOp¢2qty_seq/while/lstm_cell_55/MatMul_1/ReadVariableOp
?qty_seq/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Î
1qty_seq/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemcqty_seq_while_tensorarrayv2read_tensorlistgetitem_qty_seq_tensorarrayunstack_tensorlistfromtensor_0qty_seq_while_placeholderHqty_seq/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0­
0qty_seq/while/lstm_cell_55/MatMul/ReadVariableOpReadVariableOp;qty_seq_while_lstm_cell_55_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0Ò
!qty_seq/while/lstm_cell_55/MatMulMatMul8qty_seq/while/TensorArrayV2Read/TensorListGetItem:item:08qty_seq/while/lstm_cell_55/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
2qty_seq/while/lstm_cell_55/MatMul_1/ReadVariableOpReadVariableOp=qty_seq_while_lstm_cell_55_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0¹
#qty_seq/while/lstm_cell_55/MatMul_1MatMulqty_seq_while_placeholder_2:qty_seq/while/lstm_cell_55/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
qty_seq/while/lstm_cell_55/addAddV2+qty_seq/while/lstm_cell_55/MatMul:product:0-qty_seq/while/lstm_cell_55/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
1qty_seq/while/lstm_cell_55/BiasAdd/ReadVariableOpReadVariableOp<qty_seq_while_lstm_cell_55_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0¿
"qty_seq/while/lstm_cell_55/BiasAddBiasAdd"qty_seq/while/lstm_cell_55/add:z:09qty_seq/while/lstm_cell_55/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
*qty_seq/while/lstm_cell_55/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 qty_seq/while/lstm_cell_55/splitSplit3qty_seq/while/lstm_cell_55/split/split_dim:output:0+qty_seq/while/lstm_cell_55/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split
"qty_seq/while/lstm_cell_55/SigmoidSigmoid)qty_seq/while/lstm_cell_55/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
$qty_seq/while/lstm_cell_55/Sigmoid_1Sigmoid)qty_seq/while/lstm_cell_55/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
qty_seq/while/lstm_cell_55/mulMul(qty_seq/while/lstm_cell_55/Sigmoid_1:y:0qty_seq_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
qty_seq/while/lstm_cell_55/ReluRelu)qty_seq/while/lstm_cell_55/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ °
 qty_seq/while/lstm_cell_55/mul_1Mul&qty_seq/while/lstm_cell_55/Sigmoid:y:0-qty_seq/while/lstm_cell_55/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¥
 qty_seq/while/lstm_cell_55/add_1AddV2"qty_seq/while/lstm_cell_55/mul:z:0$qty_seq/while/lstm_cell_55/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
$qty_seq/while/lstm_cell_55/Sigmoid_2Sigmoid)qty_seq/while/lstm_cell_55/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
!qty_seq/while/lstm_cell_55/Relu_1Relu$qty_seq/while/lstm_cell_55/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ´
 qty_seq/while/lstm_cell_55/mul_2Mul(qty_seq/while/lstm_cell_55/Sigmoid_2:y:0/qty_seq/while/lstm_cell_55/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ å
2qty_seq/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemqty_seq_while_placeholder_1qty_seq_while_placeholder$qty_seq/while/lstm_cell_55/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒU
qty_seq/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
qty_seq/while/addAddV2qty_seq_while_placeholderqty_seq/while/add/y:output:0*
T0*
_output_shapes
: W
qty_seq/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
qty_seq/while/add_1AddV2(qty_seq_while_qty_seq_while_loop_counterqty_seq/while/add_1/y:output:0*
T0*
_output_shapes
: q
qty_seq/while/IdentityIdentityqty_seq/while/add_1:z:0^qty_seq/while/NoOp*
T0*
_output_shapes
: 
qty_seq/while/Identity_1Identity.qty_seq_while_qty_seq_while_maximum_iterations^qty_seq/while/NoOp*
T0*
_output_shapes
: q
qty_seq/while/Identity_2Identityqty_seq/while/add:z:0^qty_seq/while/NoOp*
T0*
_output_shapes
: ±
qty_seq/while/Identity_3IdentityBqty_seq/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^qty_seq/while/NoOp*
T0*
_output_shapes
: :éèÒ
qty_seq/while/Identity_4Identity$qty_seq/while/lstm_cell_55/mul_2:z:0^qty_seq/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
qty_seq/while/Identity_5Identity$qty_seq/while/lstm_cell_55/add_1:z:0^qty_seq/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ð
qty_seq/while/NoOpNoOp2^qty_seq/while/lstm_cell_55/BiasAdd/ReadVariableOp1^qty_seq/while/lstm_cell_55/MatMul/ReadVariableOp3^qty_seq/while/lstm_cell_55/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
qty_seq_while_identityqty_seq/while/Identity:output:0"=
qty_seq_while_identity_1!qty_seq/while/Identity_1:output:0"=
qty_seq_while_identity_2!qty_seq/while/Identity_2:output:0"=
qty_seq_while_identity_3!qty_seq/while/Identity_3:output:0"=
qty_seq_while_identity_4!qty_seq/while/Identity_4:output:0"=
qty_seq_while_identity_5!qty_seq/while/Identity_5:output:0"z
:qty_seq_while_lstm_cell_55_biasadd_readvariableop_resource<qty_seq_while_lstm_cell_55_biasadd_readvariableop_resource_0"|
;qty_seq_while_lstm_cell_55_matmul_1_readvariableop_resource=qty_seq_while_lstm_cell_55_matmul_1_readvariableop_resource_0"x
9qty_seq_while_lstm_cell_55_matmul_readvariableop_resource;qty_seq_while_lstm_cell_55_matmul_readvariableop_resource_0"P
%qty_seq_while_qty_seq_strided_slice_1'qty_seq_while_qty_seq_strided_slice_1_0"È
aqty_seq_while_tensorarrayv2read_tensorlistgetitem_qty_seq_tensorarrayunstack_tensorlistfromtensorcqty_seq_while_tensorarrayv2read_tensorlistgetitem_qty_seq_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2f
1qty_seq/while/lstm_cell_55/BiasAdd/ReadVariableOp1qty_seq/while/lstm_cell_55/BiasAdd/ReadVariableOp2d
0qty_seq/while/lstm_cell_55/MatMul/ReadVariableOp0qty_seq/while/lstm_cell_55/MatMul/ReadVariableOp2h
2qty_seq/while/lstm_cell_55/MatMul_1/ReadVariableOp2qty_seq/while/lstm_cell_55/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
È	
ö
E__inference_salt_pred_layer_call_and_return_conditional_losses_866781

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
²

D__inference_model_12_layer_call_and_return_conditional_losses_865132
inputs_0
inputs_1F
3qty_seq_lstm_cell_55_matmul_readvariableop_resource:	H
5qty_seq_lstm_cell_55_matmul_1_readvariableop_resource:	 C
4qty_seq_lstm_cell_55_biasadd_readvariableop_resource:	G
4salt_seq_lstm_cell_54_matmul_readvariableop_resource:	I
6salt_seq_lstm_cell_54_matmul_1_readvariableop_resource:	 D
5salt_seq_lstm_cell_54_biasadd_readvariableop_resource:	:
(salt_pred_matmul_readvariableop_resource:@7
)salt_pred_biasadd_readvariableop_resource:
identity¢+qty_seq/lstm_cell_55/BiasAdd/ReadVariableOp¢*qty_seq/lstm_cell_55/MatMul/ReadVariableOp¢,qty_seq/lstm_cell_55/MatMul_1/ReadVariableOp¢qty_seq/while¢ salt_pred/BiasAdd/ReadVariableOp¢salt_pred/MatMul/ReadVariableOp¢,salt_seq/lstm_cell_54/BiasAdd/ReadVariableOp¢+salt_seq/lstm_cell_54/MatMul/ReadVariableOp¢-salt_seq/lstm_cell_54/MatMul_1/ReadVariableOp¢salt_seq/whileE
qty_seq/ShapeShapeinputs_1*
T0*
_output_shapes
:e
qty_seq/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
qty_seq/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
qty_seq/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
qty_seq/strided_sliceStridedSliceqty_seq/Shape:output:0$qty_seq/strided_slice/stack:output:0&qty_seq/strided_slice/stack_1:output:0&qty_seq/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
qty_seq/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 
qty_seq/zeros/packedPackqty_seq/strided_slice:output:0qty_seq/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
qty_seq/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
qty_seq/zerosFillqty_seq/zeros/packed:output:0qty_seq/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z
qty_seq/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 
qty_seq/zeros_1/packedPackqty_seq/strided_slice:output:0!qty_seq/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
qty_seq/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
qty_seq/zeros_1Fillqty_seq/zeros_1/packed:output:0qty_seq/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ k
qty_seq/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
qty_seq/transpose	Transposeinputs_1qty_seq/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
qty_seq/Shape_1Shapeqty_seq/transpose:y:0*
T0*
_output_shapes
:g
qty_seq/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
qty_seq/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
qty_seq/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
qty_seq/strided_slice_1StridedSliceqty_seq/Shape_1:output:0&qty_seq/strided_slice_1/stack:output:0(qty_seq/strided_slice_1/stack_1:output:0(qty_seq/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#qty_seq/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÌ
qty_seq/TensorArrayV2TensorListReserve,qty_seq/TensorArrayV2/element_shape:output:0 qty_seq/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
=qty_seq/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ø
/qty_seq/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorqty_seq/transpose:y:0Fqty_seq/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒg
qty_seq/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
qty_seq/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
qty_seq/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
qty_seq/strided_slice_2StridedSliceqty_seq/transpose:y:0&qty_seq/strided_slice_2/stack:output:0(qty_seq/strided_slice_2/stack_1:output:0(qty_seq/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
*qty_seq/lstm_cell_55/MatMul/ReadVariableOpReadVariableOp3qty_seq_lstm_cell_55_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0®
qty_seq/lstm_cell_55/MatMulMatMul qty_seq/strided_slice_2:output:02qty_seq/lstm_cell_55/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
,qty_seq/lstm_cell_55/MatMul_1/ReadVariableOpReadVariableOp5qty_seq_lstm_cell_55_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0¨
qty_seq/lstm_cell_55/MatMul_1MatMulqty_seq/zeros:output:04qty_seq/lstm_cell_55/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
qty_seq/lstm_cell_55/addAddV2%qty_seq/lstm_cell_55/MatMul:product:0'qty_seq/lstm_cell_55/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+qty_seq/lstm_cell_55/BiasAdd/ReadVariableOpReadVariableOp4qty_seq_lstm_cell_55_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0­
qty_seq/lstm_cell_55/BiasAddBiasAddqty_seq/lstm_cell_55/add:z:03qty_seq/lstm_cell_55/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
$qty_seq/lstm_cell_55/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :õ
qty_seq/lstm_cell_55/splitSplit-qty_seq/lstm_cell_55/split/split_dim:output:0%qty_seq/lstm_cell_55/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split~
qty_seq/lstm_cell_55/SigmoidSigmoid#qty_seq/lstm_cell_55/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
qty_seq/lstm_cell_55/Sigmoid_1Sigmoid#qty_seq/lstm_cell_55/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
qty_seq/lstm_cell_55/mulMul"qty_seq/lstm_cell_55/Sigmoid_1:y:0qty_seq/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ x
qty_seq/lstm_cell_55/ReluRelu#qty_seq/lstm_cell_55/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
qty_seq/lstm_cell_55/mul_1Mul qty_seq/lstm_cell_55/Sigmoid:y:0'qty_seq/lstm_cell_55/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
qty_seq/lstm_cell_55/add_1AddV2qty_seq/lstm_cell_55/mul:z:0qty_seq/lstm_cell_55/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
qty_seq/lstm_cell_55/Sigmoid_2Sigmoid#qty_seq/lstm_cell_55/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ u
qty_seq/lstm_cell_55/Relu_1Reluqty_seq/lstm_cell_55/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¢
qty_seq/lstm_cell_55/mul_2Mul"qty_seq/lstm_cell_55/Sigmoid_2:y:0)qty_seq/lstm_cell_55/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ v
%qty_seq/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ð
qty_seq/TensorArrayV2_1TensorListReserve.qty_seq/TensorArrayV2_1/element_shape:output:0 qty_seq/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒN
qty_seq/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 qty_seq/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ\
qty_seq/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ò
qty_seq/whileWhile#qty_seq/while/loop_counter:output:0)qty_seq/while/maximum_iterations:output:0qty_seq/time:output:0 qty_seq/TensorArrayV2_1:handle:0qty_seq/zeros:output:0qty_seq/zeros_1:output:0 qty_seq/strided_slice_1:output:0?qty_seq/TensorArrayUnstack/TensorListFromTensor:output_handle:03qty_seq_lstm_cell_55_matmul_readvariableop_resource5qty_seq_lstm_cell_55_matmul_1_readvariableop_resource4qty_seq_lstm_cell_55_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
qty_seq_while_body_864899*%
condR
qty_seq_while_cond_864898*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
8qty_seq/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ú
*qty_seq/TensorArrayV2Stack/TensorListStackTensorListStackqty_seq/while:output:3Aqty_seq/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0p
qty_seq/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿi
qty_seq/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
qty_seq/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¯
qty_seq/strided_slice_3StridedSlice3qty_seq/TensorArrayV2Stack/TensorListStack:tensor:0&qty_seq/strided_slice_3/stack:output:0(qty_seq/strided_slice_3/stack_1:output:0(qty_seq/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maskm
qty_seq/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ®
qty_seq/transpose_1	Transpose3qty_seq/TensorArrayV2Stack/TensorListStack:tensor:0!qty_seq/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
qty_seq/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    F
salt_seq/ShapeShapeinputs_0*
T0*
_output_shapes
:f
salt_seq/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
salt_seq/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
salt_seq/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:þ
salt_seq/strided_sliceStridedSlicesalt_seq/Shape:output:0%salt_seq/strided_slice/stack:output:0'salt_seq/strided_slice/stack_1:output:0'salt_seq/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
salt_seq/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 
salt_seq/zeros/packedPacksalt_seq/strided_slice:output:0 salt_seq/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:Y
salt_seq/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
salt_seq/zerosFillsalt_seq/zeros/packed:output:0salt_seq/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [
salt_seq/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 
salt_seq/zeros_1/packedPacksalt_seq/strided_slice:output:0"salt_seq/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:[
salt_seq/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
salt_seq/zeros_1Fill salt_seq/zeros_1/packed:output:0salt_seq/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ l
salt_seq/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
salt_seq/transpose	Transposeinputs_0 salt_seq/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
salt_seq/Shape_1Shapesalt_seq/transpose:y:0*
T0*
_output_shapes
:h
salt_seq/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 salt_seq/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 salt_seq/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
salt_seq/strided_slice_1StridedSlicesalt_seq/Shape_1:output:0'salt_seq/strided_slice_1/stack:output:0)salt_seq/strided_slice_1/stack_1:output:0)salt_seq/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
$salt_seq/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÏ
salt_seq/TensorArrayV2TensorListReserve-salt_seq/TensorArrayV2/element_shape:output:0!salt_seq/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
>salt_seq/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   û
0salt_seq/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsalt_seq/transpose:y:0Gsalt_seq/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒh
salt_seq/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 salt_seq/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 salt_seq/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
salt_seq/strided_slice_2StridedSlicesalt_seq/transpose:y:0'salt_seq/strided_slice_2/stack:output:0)salt_seq/strided_slice_2/stack_1:output:0)salt_seq/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask¡
+salt_seq/lstm_cell_54/MatMul/ReadVariableOpReadVariableOp4salt_seq_lstm_cell_54_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0±
salt_seq/lstm_cell_54/MatMulMatMul!salt_seq/strided_slice_2:output:03salt_seq/lstm_cell_54/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
-salt_seq/lstm_cell_54/MatMul_1/ReadVariableOpReadVariableOp6salt_seq_lstm_cell_54_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0«
salt_seq/lstm_cell_54/MatMul_1MatMulsalt_seq/zeros:output:05salt_seq/lstm_cell_54/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
salt_seq/lstm_cell_54/addAddV2&salt_seq/lstm_cell_54/MatMul:product:0(salt_seq/lstm_cell_54/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,salt_seq/lstm_cell_54/BiasAdd/ReadVariableOpReadVariableOp5salt_seq_lstm_cell_54_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0°
salt_seq/lstm_cell_54/BiasAddBiasAddsalt_seq/lstm_cell_54/add:z:04salt_seq/lstm_cell_54/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
%salt_seq/lstm_cell_54/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ø
salt_seq/lstm_cell_54/splitSplit.salt_seq/lstm_cell_54/split/split_dim:output:0&salt_seq/lstm_cell_54/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split
salt_seq/lstm_cell_54/SigmoidSigmoid$salt_seq/lstm_cell_54/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
salt_seq/lstm_cell_54/Sigmoid_1Sigmoid$salt_seq/lstm_cell_54/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
salt_seq/lstm_cell_54/mulMul#salt_seq/lstm_cell_54/Sigmoid_1:y:0salt_seq/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z
salt_seq/lstm_cell_54/ReluRelu$salt_seq/lstm_cell_54/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¡
salt_seq/lstm_cell_54/mul_1Mul!salt_seq/lstm_cell_54/Sigmoid:y:0(salt_seq/lstm_cell_54/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
salt_seq/lstm_cell_54/add_1AddV2salt_seq/lstm_cell_54/mul:z:0salt_seq/lstm_cell_54/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
salt_seq/lstm_cell_54/Sigmoid_2Sigmoid$salt_seq/lstm_cell_54/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
salt_seq/lstm_cell_54/Relu_1Relusalt_seq/lstm_cell_54/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¥
salt_seq/lstm_cell_54/mul_2Mul#salt_seq/lstm_cell_54/Sigmoid_2:y:0*salt_seq/lstm_cell_54/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
&salt_seq/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ó
salt_seq/TensorArrayV2_1TensorListReserve/salt_seq/TensorArrayV2_1/element_shape:output:0!salt_seq/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒO
salt_seq/timeConst*
_output_shapes
: *
dtype0*
value	B : l
!salt_seq/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ]
salt_seq/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
salt_seq/whileWhile$salt_seq/while/loop_counter:output:0*salt_seq/while/maximum_iterations:output:0salt_seq/time:output:0!salt_seq/TensorArrayV2_1:handle:0salt_seq/zeros:output:0salt_seq/zeros_1:output:0!salt_seq/strided_slice_1:output:0@salt_seq/TensorArrayUnstack/TensorListFromTensor:output_handle:04salt_seq_lstm_cell_54_matmul_readvariableop_resource6salt_seq_lstm_cell_54_matmul_1_readvariableop_resource5salt_seq_lstm_cell_54_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
salt_seq_while_body_865038*&
condR
salt_seq_while_cond_865037*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
9salt_seq/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ý
+salt_seq/TensorArrayV2Stack/TensorListStackTensorListStacksalt_seq/while:output:3Bsalt_seq/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0q
salt_seq/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿj
 salt_seq/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: j
 salt_seq/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:´
salt_seq/strided_slice_3StridedSlice4salt_seq/TensorArrayV2Stack/TensorListStack:tensor:0'salt_seq/strided_slice_3/stack:output:0)salt_seq/strided_slice_3/stack_1:output:0)salt_seq/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maskn
salt_seq/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ±
salt_seq/transpose_1	Transpose4salt_seq/TensorArrayV2Stack/TensorListStack:tensor:0"salt_seq/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ d
salt_seq/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    t
salt_seq_2/IdentityIdentity!salt_seq/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
qty_seq_2/IdentityIdentity qty_seq/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ U
pattern/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :®
pattern/concatConcatV2salt_seq_2/Identity:output:0qty_seq_2/Identity:output:0pattern/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
salt_pred/MatMul/ReadVariableOpReadVariableOp(salt_pred_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
salt_pred/MatMulMatMulpattern/concat:output:0'salt_pred/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 salt_pred/BiasAdd/ReadVariableOpReadVariableOp)salt_pred_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
salt_pred/BiasAddBiasAddsalt_pred/MatMul:product:0(salt_pred/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitysalt_pred/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
NoOpNoOp,^qty_seq/lstm_cell_55/BiasAdd/ReadVariableOp+^qty_seq/lstm_cell_55/MatMul/ReadVariableOp-^qty_seq/lstm_cell_55/MatMul_1/ReadVariableOp^qty_seq/while!^salt_pred/BiasAdd/ReadVariableOp ^salt_pred/MatMul/ReadVariableOp-^salt_seq/lstm_cell_54/BiasAdd/ReadVariableOp,^salt_seq/lstm_cell_54/MatMul/ReadVariableOp.^salt_seq/lstm_cell_54/MatMul_1/ReadVariableOp^salt_seq/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2Z
+qty_seq/lstm_cell_55/BiasAdd/ReadVariableOp+qty_seq/lstm_cell_55/BiasAdd/ReadVariableOp2X
*qty_seq/lstm_cell_55/MatMul/ReadVariableOp*qty_seq/lstm_cell_55/MatMul/ReadVariableOp2\
,qty_seq/lstm_cell_55/MatMul_1/ReadVariableOp,qty_seq/lstm_cell_55/MatMul_1/ReadVariableOp2
qty_seq/whileqty_seq/while2D
 salt_pred/BiasAdd/ReadVariableOp salt_pred/BiasAdd/ReadVariableOp2B
salt_pred/MatMul/ReadVariableOpsalt_pred/MatMul/ReadVariableOp2\
,salt_seq/lstm_cell_54/BiasAdd/ReadVariableOp,salt_seq/lstm_cell_54/BiasAdd/ReadVariableOp2Z
+salt_seq/lstm_cell_54/MatMul/ReadVariableOp+salt_seq/lstm_cell_54/MatMul/ReadVariableOp2^
-salt_seq/lstm_cell_54/MatMul_1/ReadVariableOp-salt_seq/lstm_cell_54/MatMul_1/ReadVariableOp2 
salt_seq/whilesalt_seq/while:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
ØÁ

D__inference_model_12_layer_call_and_return_conditional_losses_865439
inputs_0
inputs_1F
3qty_seq_lstm_cell_55_matmul_readvariableop_resource:	H
5qty_seq_lstm_cell_55_matmul_1_readvariableop_resource:	 C
4qty_seq_lstm_cell_55_biasadd_readvariableop_resource:	G
4salt_seq_lstm_cell_54_matmul_readvariableop_resource:	I
6salt_seq_lstm_cell_54_matmul_1_readvariableop_resource:	 D
5salt_seq_lstm_cell_54_biasadd_readvariableop_resource:	:
(salt_pred_matmul_readvariableop_resource:@7
)salt_pred_biasadd_readvariableop_resource:
identity¢+qty_seq/lstm_cell_55/BiasAdd/ReadVariableOp¢*qty_seq/lstm_cell_55/MatMul/ReadVariableOp¢,qty_seq/lstm_cell_55/MatMul_1/ReadVariableOp¢qty_seq/while¢ salt_pred/BiasAdd/ReadVariableOp¢salt_pred/MatMul/ReadVariableOp¢,salt_seq/lstm_cell_54/BiasAdd/ReadVariableOp¢+salt_seq/lstm_cell_54/MatMul/ReadVariableOp¢-salt_seq/lstm_cell_54/MatMul_1/ReadVariableOp¢salt_seq/whileE
qty_seq/ShapeShapeinputs_1*
T0*
_output_shapes
:e
qty_seq/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
qty_seq/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
qty_seq/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
qty_seq/strided_sliceStridedSliceqty_seq/Shape:output:0$qty_seq/strided_slice/stack:output:0&qty_seq/strided_slice/stack_1:output:0&qty_seq/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
qty_seq/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 
qty_seq/zeros/packedPackqty_seq/strided_slice:output:0qty_seq/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
qty_seq/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
qty_seq/zerosFillqty_seq/zeros/packed:output:0qty_seq/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z
qty_seq/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 
qty_seq/zeros_1/packedPackqty_seq/strided_slice:output:0!qty_seq/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
qty_seq/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
qty_seq/zeros_1Fillqty_seq/zeros_1/packed:output:0qty_seq/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ k
qty_seq/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
qty_seq/transpose	Transposeinputs_1qty_seq/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
qty_seq/Shape_1Shapeqty_seq/transpose:y:0*
T0*
_output_shapes
:g
qty_seq/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
qty_seq/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
qty_seq/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
qty_seq/strided_slice_1StridedSliceqty_seq/Shape_1:output:0&qty_seq/strided_slice_1/stack:output:0(qty_seq/strided_slice_1/stack_1:output:0(qty_seq/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#qty_seq/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÌ
qty_seq/TensorArrayV2TensorListReserve,qty_seq/TensorArrayV2/element_shape:output:0 qty_seq/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
=qty_seq/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ø
/qty_seq/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorqty_seq/transpose:y:0Fqty_seq/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒg
qty_seq/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
qty_seq/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
qty_seq/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
qty_seq/strided_slice_2StridedSliceqty_seq/transpose:y:0&qty_seq/strided_slice_2/stack:output:0(qty_seq/strided_slice_2/stack_1:output:0(qty_seq/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
*qty_seq/lstm_cell_55/MatMul/ReadVariableOpReadVariableOp3qty_seq_lstm_cell_55_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0®
qty_seq/lstm_cell_55/MatMulMatMul qty_seq/strided_slice_2:output:02qty_seq/lstm_cell_55/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
,qty_seq/lstm_cell_55/MatMul_1/ReadVariableOpReadVariableOp5qty_seq_lstm_cell_55_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0¨
qty_seq/lstm_cell_55/MatMul_1MatMulqty_seq/zeros:output:04qty_seq/lstm_cell_55/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
qty_seq/lstm_cell_55/addAddV2%qty_seq/lstm_cell_55/MatMul:product:0'qty_seq/lstm_cell_55/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+qty_seq/lstm_cell_55/BiasAdd/ReadVariableOpReadVariableOp4qty_seq_lstm_cell_55_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0­
qty_seq/lstm_cell_55/BiasAddBiasAddqty_seq/lstm_cell_55/add:z:03qty_seq/lstm_cell_55/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
$qty_seq/lstm_cell_55/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :õ
qty_seq/lstm_cell_55/splitSplit-qty_seq/lstm_cell_55/split/split_dim:output:0%qty_seq/lstm_cell_55/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split~
qty_seq/lstm_cell_55/SigmoidSigmoid#qty_seq/lstm_cell_55/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
qty_seq/lstm_cell_55/Sigmoid_1Sigmoid#qty_seq/lstm_cell_55/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
qty_seq/lstm_cell_55/mulMul"qty_seq/lstm_cell_55/Sigmoid_1:y:0qty_seq/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ x
qty_seq/lstm_cell_55/ReluRelu#qty_seq/lstm_cell_55/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
qty_seq/lstm_cell_55/mul_1Mul qty_seq/lstm_cell_55/Sigmoid:y:0'qty_seq/lstm_cell_55/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
qty_seq/lstm_cell_55/add_1AddV2qty_seq/lstm_cell_55/mul:z:0qty_seq/lstm_cell_55/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
qty_seq/lstm_cell_55/Sigmoid_2Sigmoid#qty_seq/lstm_cell_55/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ u
qty_seq/lstm_cell_55/Relu_1Reluqty_seq/lstm_cell_55/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¢
qty_seq/lstm_cell_55/mul_2Mul"qty_seq/lstm_cell_55/Sigmoid_2:y:0)qty_seq/lstm_cell_55/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ v
%qty_seq/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ð
qty_seq/TensorArrayV2_1TensorListReserve.qty_seq/TensorArrayV2_1/element_shape:output:0 qty_seq/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒN
qty_seq/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 qty_seq/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ\
qty_seq/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ò
qty_seq/whileWhile#qty_seq/while/loop_counter:output:0)qty_seq/while/maximum_iterations:output:0qty_seq/time:output:0 qty_seq/TensorArrayV2_1:handle:0qty_seq/zeros:output:0qty_seq/zeros_1:output:0 qty_seq/strided_slice_1:output:0?qty_seq/TensorArrayUnstack/TensorListFromTensor:output_handle:03qty_seq_lstm_cell_55_matmul_readvariableop_resource5qty_seq_lstm_cell_55_matmul_1_readvariableop_resource4qty_seq_lstm_cell_55_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
qty_seq_while_body_865192*%
condR
qty_seq_while_cond_865191*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
8qty_seq/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ú
*qty_seq/TensorArrayV2Stack/TensorListStackTensorListStackqty_seq/while:output:3Aqty_seq/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0p
qty_seq/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿi
qty_seq/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
qty_seq/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¯
qty_seq/strided_slice_3StridedSlice3qty_seq/TensorArrayV2Stack/TensorListStack:tensor:0&qty_seq/strided_slice_3/stack:output:0(qty_seq/strided_slice_3/stack_1:output:0(qty_seq/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maskm
qty_seq/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ®
qty_seq/transpose_1	Transpose3qty_seq/TensorArrayV2Stack/TensorListStack:tensor:0!qty_seq/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
qty_seq/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    F
salt_seq/ShapeShapeinputs_0*
T0*
_output_shapes
:f
salt_seq/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
salt_seq/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
salt_seq/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:þ
salt_seq/strided_sliceStridedSlicesalt_seq/Shape:output:0%salt_seq/strided_slice/stack:output:0'salt_seq/strided_slice/stack_1:output:0'salt_seq/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
salt_seq/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 
salt_seq/zeros/packedPacksalt_seq/strided_slice:output:0 salt_seq/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:Y
salt_seq/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
salt_seq/zerosFillsalt_seq/zeros/packed:output:0salt_seq/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [
salt_seq/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 
salt_seq/zeros_1/packedPacksalt_seq/strided_slice:output:0"salt_seq/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:[
salt_seq/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
salt_seq/zeros_1Fill salt_seq/zeros_1/packed:output:0salt_seq/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ l
salt_seq/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
salt_seq/transpose	Transposeinputs_0 salt_seq/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
salt_seq/Shape_1Shapesalt_seq/transpose:y:0*
T0*
_output_shapes
:h
salt_seq/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 salt_seq/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 salt_seq/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
salt_seq/strided_slice_1StridedSlicesalt_seq/Shape_1:output:0'salt_seq/strided_slice_1/stack:output:0)salt_seq/strided_slice_1/stack_1:output:0)salt_seq/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
$salt_seq/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÏ
salt_seq/TensorArrayV2TensorListReserve-salt_seq/TensorArrayV2/element_shape:output:0!salt_seq/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
>salt_seq/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   û
0salt_seq/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsalt_seq/transpose:y:0Gsalt_seq/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒh
salt_seq/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 salt_seq/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 salt_seq/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
salt_seq/strided_slice_2StridedSlicesalt_seq/transpose:y:0'salt_seq/strided_slice_2/stack:output:0)salt_seq/strided_slice_2/stack_1:output:0)salt_seq/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask¡
+salt_seq/lstm_cell_54/MatMul/ReadVariableOpReadVariableOp4salt_seq_lstm_cell_54_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0±
salt_seq/lstm_cell_54/MatMulMatMul!salt_seq/strided_slice_2:output:03salt_seq/lstm_cell_54/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
-salt_seq/lstm_cell_54/MatMul_1/ReadVariableOpReadVariableOp6salt_seq_lstm_cell_54_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0«
salt_seq/lstm_cell_54/MatMul_1MatMulsalt_seq/zeros:output:05salt_seq/lstm_cell_54/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
salt_seq/lstm_cell_54/addAddV2&salt_seq/lstm_cell_54/MatMul:product:0(salt_seq/lstm_cell_54/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,salt_seq/lstm_cell_54/BiasAdd/ReadVariableOpReadVariableOp5salt_seq_lstm_cell_54_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0°
salt_seq/lstm_cell_54/BiasAddBiasAddsalt_seq/lstm_cell_54/add:z:04salt_seq/lstm_cell_54/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
%salt_seq/lstm_cell_54/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ø
salt_seq/lstm_cell_54/splitSplit.salt_seq/lstm_cell_54/split/split_dim:output:0&salt_seq/lstm_cell_54/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split
salt_seq/lstm_cell_54/SigmoidSigmoid$salt_seq/lstm_cell_54/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
salt_seq/lstm_cell_54/Sigmoid_1Sigmoid$salt_seq/lstm_cell_54/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
salt_seq/lstm_cell_54/mulMul#salt_seq/lstm_cell_54/Sigmoid_1:y:0salt_seq/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z
salt_seq/lstm_cell_54/ReluRelu$salt_seq/lstm_cell_54/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¡
salt_seq/lstm_cell_54/mul_1Mul!salt_seq/lstm_cell_54/Sigmoid:y:0(salt_seq/lstm_cell_54/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
salt_seq/lstm_cell_54/add_1AddV2salt_seq/lstm_cell_54/mul:z:0salt_seq/lstm_cell_54/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
salt_seq/lstm_cell_54/Sigmoid_2Sigmoid$salt_seq/lstm_cell_54/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
salt_seq/lstm_cell_54/Relu_1Relusalt_seq/lstm_cell_54/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¥
salt_seq/lstm_cell_54/mul_2Mul#salt_seq/lstm_cell_54/Sigmoid_2:y:0*salt_seq/lstm_cell_54/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
&salt_seq/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ó
salt_seq/TensorArrayV2_1TensorListReserve/salt_seq/TensorArrayV2_1/element_shape:output:0!salt_seq/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒO
salt_seq/timeConst*
_output_shapes
: *
dtype0*
value	B : l
!salt_seq/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ]
salt_seq/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
salt_seq/whileWhile$salt_seq/while/loop_counter:output:0*salt_seq/while/maximum_iterations:output:0salt_seq/time:output:0!salt_seq/TensorArrayV2_1:handle:0salt_seq/zeros:output:0salt_seq/zeros_1:output:0!salt_seq/strided_slice_1:output:0@salt_seq/TensorArrayUnstack/TensorListFromTensor:output_handle:04salt_seq_lstm_cell_54_matmul_readvariableop_resource6salt_seq_lstm_cell_54_matmul_1_readvariableop_resource5salt_seq_lstm_cell_54_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
salt_seq_while_body_865331*&
condR
salt_seq_while_cond_865330*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
9salt_seq/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ý
+salt_seq/TensorArrayV2Stack/TensorListStackTensorListStacksalt_seq/while:output:3Bsalt_seq/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0q
salt_seq/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿj
 salt_seq/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: j
 salt_seq/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:´
salt_seq/strided_slice_3StridedSlice4salt_seq/TensorArrayV2Stack/TensorListStack:tensor:0'salt_seq/strided_slice_3/stack:output:0)salt_seq/strided_slice_3/stack_1:output:0)salt_seq/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maskn
salt_seq/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ±
salt_seq/transpose_1	Transpose4salt_seq/TensorArrayV2Stack/TensorListStack:tensor:0"salt_seq/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ d
salt_seq/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ]
salt_seq_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?
salt_seq_2/dropout/MulMul!salt_seq/strided_slice_3:output:0!salt_seq_2/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ i
salt_seq_2/dropout/ShapeShape!salt_seq/strided_slice_3:output:0*
T0*
_output_shapes
:®
/salt_seq_2/dropout/random_uniform/RandomUniformRandomUniform!salt_seq_2/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedf
!salt_seq_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>Ç
salt_seq_2/dropout/GreaterEqualGreaterEqual8salt_seq_2/dropout/random_uniform/RandomUniform:output:0*salt_seq_2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
salt_seq_2/dropout/CastCast#salt_seq_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
salt_seq_2/dropout/Mul_1Mulsalt_seq_2/dropout/Mul:z:0salt_seq_2/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ \
qty_seq_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?
qty_seq_2/dropout/MulMul qty_seq/strided_slice_3:output:0 qty_seq_2/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ g
qty_seq_2/dropout/ShapeShape qty_seq/strided_slice_3:output:0*
T0*
_output_shapes
:¹
.qty_seq_2/dropout/random_uniform/RandomUniformRandomUniform qty_seq_2/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seed*
seed2e
 qty_seq_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>Ä
qty_seq_2/dropout/GreaterEqualGreaterEqual7qty_seq_2/dropout/random_uniform/RandomUniform:output:0)qty_seq_2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
qty_seq_2/dropout/CastCast"qty_seq_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
qty_seq_2/dropout/Mul_1Mulqty_seq_2/dropout/Mul:z:0qty_seq_2/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ U
pattern/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :®
pattern/concatConcatV2salt_seq_2/dropout/Mul_1:z:0qty_seq_2/dropout/Mul_1:z:0pattern/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
salt_pred/MatMul/ReadVariableOpReadVariableOp(salt_pred_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
salt_pred/MatMulMatMulpattern/concat:output:0'salt_pred/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 salt_pred/BiasAdd/ReadVariableOpReadVariableOp)salt_pred_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
salt_pred/BiasAddBiasAddsalt_pred/MatMul:product:0(salt_pred/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitysalt_pred/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
NoOpNoOp,^qty_seq/lstm_cell_55/BiasAdd/ReadVariableOp+^qty_seq/lstm_cell_55/MatMul/ReadVariableOp-^qty_seq/lstm_cell_55/MatMul_1/ReadVariableOp^qty_seq/while!^salt_pred/BiasAdd/ReadVariableOp ^salt_pred/MatMul/ReadVariableOp-^salt_seq/lstm_cell_54/BiasAdd/ReadVariableOp,^salt_seq/lstm_cell_54/MatMul/ReadVariableOp.^salt_seq/lstm_cell_54/MatMul_1/ReadVariableOp^salt_seq/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2Z
+qty_seq/lstm_cell_55/BiasAdd/ReadVariableOp+qty_seq/lstm_cell_55/BiasAdd/ReadVariableOp2X
*qty_seq/lstm_cell_55/MatMul/ReadVariableOp*qty_seq/lstm_cell_55/MatMul/ReadVariableOp2\
,qty_seq/lstm_cell_55/MatMul_1/ReadVariableOp,qty_seq/lstm_cell_55/MatMul_1/ReadVariableOp2
qty_seq/whileqty_seq/while2D
 salt_pred/BiasAdd/ReadVariableOp salt_pred/BiasAdd/ReadVariableOp2B
salt_pred/MatMul/ReadVariableOpsalt_pred/MatMul/ReadVariableOp2\
,salt_seq/lstm_cell_54/BiasAdd/ReadVariableOp,salt_seq/lstm_cell_54/BiasAdd/ReadVariableOp2Z
+salt_seq/lstm_cell_54/MatMul/ReadVariableOp+salt_seq/lstm_cell_54/MatMul/ReadVariableOp2^
-salt_seq/lstm_cell_54/MatMul_1/ReadVariableOp-salt_seq/lstm_cell_54/MatMul_1/ReadVariableOp2 
salt_seq/whilesalt_seq/while:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
µ
Ã
while_cond_866181
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_866181___redundant_placeholder04
0while_while_cond_866181___redundant_placeholder14
0while_while_cond_866181___redundant_placeholder24
0while_while_cond_866181___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
¤L
ð
"model_12_qty_seq_while_body_862948>
:model_12_qty_seq_while_model_12_qty_seq_while_loop_counterD
@model_12_qty_seq_while_model_12_qty_seq_while_maximum_iterations&
"model_12_qty_seq_while_placeholder(
$model_12_qty_seq_while_placeholder_1(
$model_12_qty_seq_while_placeholder_2(
$model_12_qty_seq_while_placeholder_3=
9model_12_qty_seq_while_model_12_qty_seq_strided_slice_1_0y
umodel_12_qty_seq_while_tensorarrayv2read_tensorlistgetitem_model_12_qty_seq_tensorarrayunstack_tensorlistfromtensor_0W
Dmodel_12_qty_seq_while_lstm_cell_55_matmul_readvariableop_resource_0:	Y
Fmodel_12_qty_seq_while_lstm_cell_55_matmul_1_readvariableop_resource_0:	 T
Emodel_12_qty_seq_while_lstm_cell_55_biasadd_readvariableop_resource_0:	#
model_12_qty_seq_while_identity%
!model_12_qty_seq_while_identity_1%
!model_12_qty_seq_while_identity_2%
!model_12_qty_seq_while_identity_3%
!model_12_qty_seq_while_identity_4%
!model_12_qty_seq_while_identity_5;
7model_12_qty_seq_while_model_12_qty_seq_strided_slice_1w
smodel_12_qty_seq_while_tensorarrayv2read_tensorlistgetitem_model_12_qty_seq_tensorarrayunstack_tensorlistfromtensorU
Bmodel_12_qty_seq_while_lstm_cell_55_matmul_readvariableop_resource:	W
Dmodel_12_qty_seq_while_lstm_cell_55_matmul_1_readvariableop_resource:	 R
Cmodel_12_qty_seq_while_lstm_cell_55_biasadd_readvariableop_resource:	¢:model_12/qty_seq/while/lstm_cell_55/BiasAdd/ReadVariableOp¢9model_12/qty_seq/while/lstm_cell_55/MatMul/ReadVariableOp¢;model_12/qty_seq/while/lstm_cell_55/MatMul_1/ReadVariableOp
Hmodel_12/qty_seq/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   û
:model_12/qty_seq/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemumodel_12_qty_seq_while_tensorarrayv2read_tensorlistgetitem_model_12_qty_seq_tensorarrayunstack_tensorlistfromtensor_0"model_12_qty_seq_while_placeholderQmodel_12/qty_seq/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0¿
9model_12/qty_seq/while/lstm_cell_55/MatMul/ReadVariableOpReadVariableOpDmodel_12_qty_seq_while_lstm_cell_55_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0í
*model_12/qty_seq/while/lstm_cell_55/MatMulMatMulAmodel_12/qty_seq/while/TensorArrayV2Read/TensorListGetItem:item:0Amodel_12/qty_seq/while/lstm_cell_55/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
;model_12/qty_seq/while/lstm_cell_55/MatMul_1/ReadVariableOpReadVariableOpFmodel_12_qty_seq_while_lstm_cell_55_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0Ô
,model_12/qty_seq/while/lstm_cell_55/MatMul_1MatMul$model_12_qty_seq_while_placeholder_2Cmodel_12/qty_seq/while/lstm_cell_55/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
'model_12/qty_seq/while/lstm_cell_55/addAddV24model_12/qty_seq/while/lstm_cell_55/MatMul:product:06model_12/qty_seq/while/lstm_cell_55/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
:model_12/qty_seq/while/lstm_cell_55/BiasAdd/ReadVariableOpReadVariableOpEmodel_12_qty_seq_while_lstm_cell_55_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0Ú
+model_12/qty_seq/while/lstm_cell_55/BiasAddBiasAdd+model_12/qty_seq/while/lstm_cell_55/add:z:0Bmodel_12/qty_seq/while/lstm_cell_55/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
3model_12/qty_seq/while/lstm_cell_55/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¢
)model_12/qty_seq/while/lstm_cell_55/splitSplit<model_12/qty_seq/while/lstm_cell_55/split/split_dim:output:04model_12/qty_seq/while/lstm_cell_55/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split
+model_12/qty_seq/while/lstm_cell_55/SigmoidSigmoid2model_12/qty_seq/while/lstm_cell_55/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
-model_12/qty_seq/while/lstm_cell_55/Sigmoid_1Sigmoid2model_12/qty_seq/while/lstm_cell_55/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¹
'model_12/qty_seq/while/lstm_cell_55/mulMul1model_12/qty_seq/while/lstm_cell_55/Sigmoid_1:y:0$model_12_qty_seq_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(model_12/qty_seq/while/lstm_cell_55/ReluRelu2model_12/qty_seq/while/lstm_cell_55/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ë
)model_12/qty_seq/while/lstm_cell_55/mul_1Mul/model_12/qty_seq/while/lstm_cell_55/Sigmoid:y:06model_12/qty_seq/while/lstm_cell_55/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ À
)model_12/qty_seq/while/lstm_cell_55/add_1AddV2+model_12/qty_seq/while/lstm_cell_55/mul:z:0-model_12/qty_seq/while/lstm_cell_55/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
-model_12/qty_seq/while/lstm_cell_55/Sigmoid_2Sigmoid2model_12/qty_seq/while/lstm_cell_55/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
*model_12/qty_seq/while/lstm_cell_55/Relu_1Relu-model_12/qty_seq/while/lstm_cell_55/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ï
)model_12/qty_seq/while/lstm_cell_55/mul_2Mul1model_12/qty_seq/while/lstm_cell_55/Sigmoid_2:y:08model_12/qty_seq/while/lstm_cell_55/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
;model_12/qty_seq/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$model_12_qty_seq_while_placeholder_1"model_12_qty_seq_while_placeholder-model_12/qty_seq/while/lstm_cell_55/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒ^
model_12/qty_seq/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
model_12/qty_seq/while/addAddV2"model_12_qty_seq_while_placeholder%model_12/qty_seq/while/add/y:output:0*
T0*
_output_shapes
: `
model_12/qty_seq/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :«
model_12/qty_seq/while/add_1AddV2:model_12_qty_seq_while_model_12_qty_seq_while_loop_counter'model_12/qty_seq/while/add_1/y:output:0*
T0*
_output_shapes
: 
model_12/qty_seq/while/IdentityIdentity model_12/qty_seq/while/add_1:z:0^model_12/qty_seq/while/NoOp*
T0*
_output_shapes
: ®
!model_12/qty_seq/while/Identity_1Identity@model_12_qty_seq_while_model_12_qty_seq_while_maximum_iterations^model_12/qty_seq/while/NoOp*
T0*
_output_shapes
: 
!model_12/qty_seq/while/Identity_2Identitymodel_12/qty_seq/while/add:z:0^model_12/qty_seq/while/NoOp*
T0*
_output_shapes
: Ì
!model_12/qty_seq/while/Identity_3IdentityKmodel_12/qty_seq/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^model_12/qty_seq/while/NoOp*
T0*
_output_shapes
: :éèÒ¬
!model_12/qty_seq/while/Identity_4Identity-model_12/qty_seq/while/lstm_cell_55/mul_2:z:0^model_12/qty_seq/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¬
!model_12/qty_seq/while/Identity_5Identity-model_12/qty_seq/while/lstm_cell_55/add_1:z:0^model_12/qty_seq/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
model_12/qty_seq/while/NoOpNoOp;^model_12/qty_seq/while/lstm_cell_55/BiasAdd/ReadVariableOp:^model_12/qty_seq/while/lstm_cell_55/MatMul/ReadVariableOp<^model_12/qty_seq/while/lstm_cell_55/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "K
model_12_qty_seq_while_identity(model_12/qty_seq/while/Identity:output:0"O
!model_12_qty_seq_while_identity_1*model_12/qty_seq/while/Identity_1:output:0"O
!model_12_qty_seq_while_identity_2*model_12/qty_seq/while/Identity_2:output:0"O
!model_12_qty_seq_while_identity_3*model_12/qty_seq/while/Identity_3:output:0"O
!model_12_qty_seq_while_identity_4*model_12/qty_seq/while/Identity_4:output:0"O
!model_12_qty_seq_while_identity_5*model_12/qty_seq/while/Identity_5:output:0"
Cmodel_12_qty_seq_while_lstm_cell_55_biasadd_readvariableop_resourceEmodel_12_qty_seq_while_lstm_cell_55_biasadd_readvariableop_resource_0"
Dmodel_12_qty_seq_while_lstm_cell_55_matmul_1_readvariableop_resourceFmodel_12_qty_seq_while_lstm_cell_55_matmul_1_readvariableop_resource_0"
Bmodel_12_qty_seq_while_lstm_cell_55_matmul_readvariableop_resourceDmodel_12_qty_seq_while_lstm_cell_55_matmul_readvariableop_resource_0"t
7model_12_qty_seq_while_model_12_qty_seq_strided_slice_19model_12_qty_seq_while_model_12_qty_seq_strided_slice_1_0"ì
smodel_12_qty_seq_while_tensorarrayv2read_tensorlistgetitem_model_12_qty_seq_tensorarrayunstack_tensorlistfromtensorumodel_12_qty_seq_while_tensorarrayv2read_tensorlistgetitem_model_12_qty_seq_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2x
:model_12/qty_seq/while/lstm_cell_55/BiasAdd/ReadVariableOp:model_12/qty_seq/while/lstm_cell_55/BiasAdd/ReadVariableOp2v
9model_12/qty_seq/while/lstm_cell_55/MatMul/ReadVariableOp9model_12/qty_seq/while/lstm_cell_55/MatMul/ReadVariableOp2z
;model_12/qty_seq/while/lstm_cell_55/MatMul_1/ReadVariableOp;model_12/qty_seq/while/lstm_cell_55/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
¡
G
+__inference_salt_seq_2_layer_call_fn_866700

inputs
identity±
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_salt_seq_2_layer_call_and_return_conditional_losses_864195`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ø
c
E__inference_qty_seq_2_layer_call_and_return_conditional_losses_864202

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
´
m
C__inference_pattern_layer_call_and_return_conditional_losses_864211

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :u
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
©J

D__inference_salt_seq_layer_call_and_return_conditional_losses_865936

inputs>
+lstm_cell_54_matmul_readvariableop_resource:	@
-lstm_cell_54_matmul_1_readvariableop_resource:	 ;
,lstm_cell_54_biasadd_readvariableop_resource:	
identity¢#lstm_cell_54/BiasAdd/ReadVariableOp¢"lstm_cell_54/MatMul/ReadVariableOp¢$lstm_cell_54/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_54/MatMul/ReadVariableOpReadVariableOp+lstm_cell_54_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_54/MatMulMatMulstrided_slice_2:output:0*lstm_cell_54/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_54/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_54_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0
lstm_cell_54/MatMul_1MatMulzeros:output:0,lstm_cell_54/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_54/addAddV2lstm_cell_54/MatMul:product:0lstm_cell_54/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_54/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_54_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_54/BiasAddBiasAddlstm_cell_54/add:z:0+lstm_cell_54/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_54/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstm_cell_54/splitSplit%lstm_cell_54/split/split_dim:output:0lstm_cell_54/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitn
lstm_cell_54/SigmoidSigmoidlstm_cell_54/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
lstm_cell_54/Sigmoid_1Sigmoidlstm_cell_54/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
lstm_cell_54/mulMullstm_cell_54/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ h
lstm_cell_54/ReluRelulstm_cell_54/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_54/mul_1Mullstm_cell_54/Sigmoid:y:0lstm_cell_54/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ {
lstm_cell_54/add_1AddV2lstm_cell_54/mul:z:0lstm_cell_54/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
lstm_cell_54/Sigmoid_2Sigmoidlstm_cell_54/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ e
lstm_cell_54/Relu_1Relulstm_cell_54/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_54/mul_2Mullstm_cell_54/Sigmoid_2:y:0!lstm_cell_54/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_54_matmul_readvariableop_resource-lstm_cell_54_matmul_1_readvariableop_resource,lstm_cell_54_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_865852*
condR
while_cond_865851*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ À
NoOpNoOp$^lstm_cell_54/BiasAdd/ReadVariableOp#^lstm_cell_54/MatMul/ReadVariableOp%^lstm_cell_54/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_54/BiasAdd/ReadVariableOp#lstm_cell_54/BiasAdd/ReadVariableOp2H
"lstm_cell_54/MatMul/ReadVariableOp"lstm_cell_54/MatMul/ReadVariableOp2L
$lstm_cell_54/MatMul_1/ReadVariableOp$lstm_cell_54/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¼
o
C__inference_pattern_layer_call_and_return_conditional_losses_866762
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :w
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/1
¨J

C__inference_qty_seq_layer_call_and_return_conditional_losses_866695

inputs>
+lstm_cell_55_matmul_readvariableop_resource:	@
-lstm_cell_55_matmul_1_readvariableop_resource:	 ;
,lstm_cell_55_biasadd_readvariableop_resource:	
identity¢#lstm_cell_55/BiasAdd/ReadVariableOp¢"lstm_cell_55/MatMul/ReadVariableOp¢$lstm_cell_55/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_55/MatMul/ReadVariableOpReadVariableOp+lstm_cell_55_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_55/MatMulMatMulstrided_slice_2:output:0*lstm_cell_55/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_55/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_55_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0
lstm_cell_55/MatMul_1MatMulzeros:output:0,lstm_cell_55/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_55/addAddV2lstm_cell_55/MatMul:product:0lstm_cell_55/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_55/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_55_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_55/BiasAddBiasAddlstm_cell_55/add:z:0+lstm_cell_55/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_55/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstm_cell_55/splitSplit%lstm_cell_55/split/split_dim:output:0lstm_cell_55/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitn
lstm_cell_55/SigmoidSigmoidlstm_cell_55/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
lstm_cell_55/Sigmoid_1Sigmoidlstm_cell_55/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
lstm_cell_55/mulMullstm_cell_55/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ h
lstm_cell_55/ReluRelulstm_cell_55/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_55/mul_1Mullstm_cell_55/Sigmoid:y:0lstm_cell_55/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ {
lstm_cell_55/add_1AddV2lstm_cell_55/mul:z:0lstm_cell_55/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
lstm_cell_55/Sigmoid_2Sigmoidlstm_cell_55/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ e
lstm_cell_55/Relu_1Relulstm_cell_55/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_55/mul_2Mullstm_cell_55/Sigmoid_2:y:0!lstm_cell_55/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_55_matmul_readvariableop_resource-lstm_cell_55_matmul_1_readvariableop_resource,lstm_cell_55_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_866611*
condR
while_cond_866610*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ À
NoOpNoOp$^lstm_cell_55/BiasAdd/ReadVariableOp#^lstm_cell_55/MatMul/ReadVariableOp%^lstm_cell_55/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_55/BiasAdd/ReadVariableOp#lstm_cell_55/BiasAdd/ReadVariableOp2H
"lstm_cell_55/MatMul/ReadVariableOp"lstm_cell_55/MatMul/ReadVariableOp2L
$lstm_cell_55/MatMul_1/ReadVariableOp$lstm_cell_55/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ù"
ã
while_body_863262
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_54_863286_0:	.
while_lstm_cell_54_863288_0:	 *
while_lstm_cell_54_863290_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_54_863286:	,
while_lstm_cell_54_863288:	 (
while_lstm_cell_54_863290:	¢*while/lstm_cell_54/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0³
*while/lstm_cell_54/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_54_863286_0while_lstm_cell_54_863288_0while_lstm_cell_54_863290_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_54_layer_call_and_return_conditional_losses_863248Ü
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_54/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity3while/lstm_cell_54/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/Identity_5Identity3while/lstm_cell_54/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ y

while/NoOpNoOp+^while/lstm_cell_54/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_54_863286while_lstm_cell_54_863286_0"8
while_lstm_cell_54_863288while_lstm_cell_54_863288_0"8
while_lstm_cell_54_863290while_lstm_cell_54_863290_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2X
*while/lstm_cell_54/StatefulPartitionedCall*while/lstm_cell_54/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 

·
(__inference_qty_seq_layer_call_fn_866090
inputs_0
unknown:	
	unknown_0:	 
	unknown_1:	
identity¢StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_qty_seq_layer_call_and_return_conditional_losses_863681o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
ù"
ã
while_body_863453
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_54_863477_0:	.
while_lstm_cell_54_863479_0:	 *
while_lstm_cell_54_863481_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_54_863477:	,
while_lstm_cell_54_863479:	 (
while_lstm_cell_54_863481:	¢*while/lstm_cell_54/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0³
*while/lstm_cell_54/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_54_863477_0while_lstm_cell_54_863479_0while_lstm_cell_54_863481_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_54_layer_call_and_return_conditional_losses_863394Ü
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_54/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity3while/lstm_cell_54/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/Identity_5Identity3while/lstm_cell_54/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ y

while/NoOpNoOp+^while/lstm_cell_54/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_54_863477while_lstm_cell_54_863477_0"8
while_lstm_cell_54_863479while_lstm_cell_54_863479_0"8
while_lstm_cell_54_863481while_lstm_cell_54_863481_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2X
*while/lstm_cell_54/StatefulPartitionedCall*while/lstm_cell_54/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
ù
¶
)__inference_salt_seq_layer_call_fn_865496

inputs
unknown:	
	unknown_0:	 
	unknown_1:	
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_salt_seq_layer_call_and_return_conditional_losses_864182o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ë
D__inference_model_12_layer_call_and_return_conditional_losses_864230

inputs
inputs_1!
qty_seq_864033:	!
qty_seq_864035:	 
qty_seq_864037:	"
salt_seq_864183:	"
salt_seq_864185:	 
salt_seq_864187:	"
salt_pred_864224:@
salt_pred_864226:
identity¢qty_seq/StatefulPartitionedCall¢!salt_pred/StatefulPartitionedCall¢ salt_seq/StatefulPartitionedCall
qty_seq/StatefulPartitionedCallStatefulPartitionedCallinputs_1qty_seq_864033qty_seq_864035qty_seq_864037*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_qty_seq_layer_call_and_return_conditional_losses_864032
 salt_seq/StatefulPartitionedCallStatefulPartitionedCallinputssalt_seq_864183salt_seq_864185salt_seq_864187*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_salt_seq_layer_call_and_return_conditional_losses_864182ß
salt_seq_2/PartitionedCallPartitionedCall)salt_seq/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_salt_seq_2_layer_call_and_return_conditional_losses_864195Ü
qty_seq_2/PartitionedCallPartitionedCall(qty_seq/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_qty_seq_2_layer_call_and_return_conditional_losses_864202ø
pattern/PartitionedCallPartitionedCall#salt_seq_2/PartitionedCall:output:0"qty_seq_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_pattern_layer_call_and_return_conditional_losses_864211
!salt_pred/StatefulPartitionedCallStatefulPartitionedCall pattern/PartitionedCall:output:0salt_pred_864224salt_pred_864226*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_salt_pred_layer_call_and_return_conditional_losses_864223y
IdentityIdentity*salt_pred/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯
NoOpNoOp ^qty_seq/StatefulPartitionedCall"^salt_pred/StatefulPartitionedCall!^salt_seq/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2B
qty_seq/StatefulPartitionedCallqty_seq/StatefulPartitionedCall2F
!salt_pred/StatefulPartitionedCall!salt_pred/StatefulPartitionedCall2D
 salt_seq/StatefulPartitionedCall salt_seq/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:SO
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ

H__inference_lstm_cell_55_layer_call_and_return_conditional_losses_863744

inputs

states
states_11
matmul_readvariableop_resource:	3
 matmul_1_readvariableop_resource:	 .
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ N
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_namestates
8

C__inference_qty_seq_layer_call_and_return_conditional_losses_863872

inputs&
lstm_cell_55_863790:	&
lstm_cell_55_863792:	 "
lstm_cell_55_863794:	
identity¢$lstm_cell_55/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskõ
$lstm_cell_55/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_55_863790lstm_cell_55_863792lstm_cell_55_863794*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_55_layer_call_and_return_conditional_losses_863744n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ·
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_55_863790lstm_cell_55_863792lstm_cell_55_863794*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_863803*
condR
while_cond_863802*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ u
NoOpNoOp%^lstm_cell_55/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2L
$lstm_cell_55/StatefulPartitionedCall$lstm_cell_55/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã

"__inference__traced_restore_867197
file_prefix3
!assignvariableop_salt_pred_kernel:@/
!assignvariableop_1_salt_pred_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: B
/assignvariableop_7_salt_seq_lstm_cell_54_kernel:	L
9assignvariableop_8_salt_seq_lstm_cell_54_recurrent_kernel:	 <
-assignvariableop_9_salt_seq_lstm_cell_54_bias:	B
/assignvariableop_10_qty_seq_lstm_cell_55_kernel:	L
9assignvariableop_11_qty_seq_lstm_cell_55_recurrent_kernel:	 <
-assignvariableop_12_qty_seq_lstm_cell_55_bias:	#
assignvariableop_13_total: #
assignvariableop_14_count: =
+assignvariableop_15_adam_salt_pred_kernel_m:@7
)assignvariableop_16_adam_salt_pred_bias_m:J
7assignvariableop_17_adam_salt_seq_lstm_cell_54_kernel_m:	T
Aassignvariableop_18_adam_salt_seq_lstm_cell_54_recurrent_kernel_m:	 D
5assignvariableop_19_adam_salt_seq_lstm_cell_54_bias_m:	I
6assignvariableop_20_adam_qty_seq_lstm_cell_55_kernel_m:	S
@assignvariableop_21_adam_qty_seq_lstm_cell_55_recurrent_kernel_m:	 C
4assignvariableop_22_adam_qty_seq_lstm_cell_55_bias_m:	=
+assignvariableop_23_adam_salt_pred_kernel_v:@7
)assignvariableop_24_adam_salt_pred_bias_v:J
7assignvariableop_25_adam_salt_seq_lstm_cell_54_kernel_v:	T
Aassignvariableop_26_adam_salt_seq_lstm_cell_54_recurrent_kernel_v:	 D
5assignvariableop_27_adam_salt_seq_lstm_cell_54_bias_v:	I
6assignvariableop_28_adam_qty_seq_lstm_cell_55_kernel_v:	S
@assignvariableop_29_adam_qty_seq_lstm_cell_55_recurrent_kernel_v:	 C
4assignvariableop_30_adam_qty_seq_lstm_cell_55_bias_v:	
identity_32¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9¸
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*Þ
valueÔBÑ B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH°
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Á
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
::::::::::::::::::::::::::::::::*.
dtypes$
"2 	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp!assignvariableop_salt_pred_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp!assignvariableop_1_salt_pred_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp/assignvariableop_7_salt_seq_lstm_cell_54_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_8AssignVariableOp9assignvariableop_8_salt_seq_lstm_cell_54_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp-assignvariableop_9_salt_seq_lstm_cell_54_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_10AssignVariableOp/assignvariableop_10_qty_seq_lstm_cell_55_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_11AssignVariableOp9assignvariableop_11_qty_seq_lstm_cell_55_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp-assignvariableop_12_qty_seq_lstm_cell_55_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp+assignvariableop_15_adam_salt_pred_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp)assignvariableop_16_adam_salt_pred_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_17AssignVariableOp7assignvariableop_17_adam_salt_seq_lstm_cell_54_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_18AssignVariableOpAassignvariableop_18_adam_salt_seq_lstm_cell_54_recurrent_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_19AssignVariableOp5assignvariableop_19_adam_salt_seq_lstm_cell_54_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_20AssignVariableOp6assignvariableop_20_adam_qty_seq_lstm_cell_55_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_21AssignVariableOp@assignvariableop_21_adam_qty_seq_lstm_cell_55_recurrent_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_22AssignVariableOp4assignvariableop_22_adam_qty_seq_lstm_cell_55_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_salt_pred_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_salt_pred_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_25AssignVariableOp7assignvariableop_25_adam_salt_seq_lstm_cell_54_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_26AssignVariableOpAassignvariableop_26_adam_salt_seq_lstm_cell_54_recurrent_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_27AssignVariableOp5assignvariableop_27_adam_salt_seq_lstm_cell_54_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_28AssignVariableOp6assignvariableop_28_adam_qty_seq_lstm_cell_55_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_29AssignVariableOp@assignvariableop_29_adam_qty_seq_lstm_cell_55_recurrent_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_30AssignVariableOp4assignvariableop_30_adam_qty_seq_lstm_cell_55_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ù
Identity_31Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_32IdentityIdentity_31:output:0^NoOp_1*
T0*
_output_shapes
: æ
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_32Identity_32:output:0*S
_input_shapesB
@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
µ
Ã
while_cond_865994
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_865994___redundant_placeholder04
0while_while_cond_865994___redundant_placeholder14
0while_while_cond_865994___redundant_placeholder24
0while_while_cond_865994___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
ÞJ

D__inference_salt_seq_layer_call_and_return_conditional_losses_865793
inputs_0>
+lstm_cell_54_matmul_readvariableop_resource:	@
-lstm_cell_54_matmul_1_readvariableop_resource:	 ;
,lstm_cell_54_biasadd_readvariableop_resource:	
identity¢#lstm_cell_54/BiasAdd/ReadVariableOp¢"lstm_cell_54/MatMul/ReadVariableOp¢$lstm_cell_54/MatMul_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_54/MatMul/ReadVariableOpReadVariableOp+lstm_cell_54_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_54/MatMulMatMulstrided_slice_2:output:0*lstm_cell_54/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_54/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_54_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0
lstm_cell_54/MatMul_1MatMulzeros:output:0,lstm_cell_54/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_54/addAddV2lstm_cell_54/MatMul:product:0lstm_cell_54/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_54/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_54_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_54/BiasAddBiasAddlstm_cell_54/add:z:0+lstm_cell_54/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_54/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstm_cell_54/splitSplit%lstm_cell_54/split/split_dim:output:0lstm_cell_54/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitn
lstm_cell_54/SigmoidSigmoidlstm_cell_54/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
lstm_cell_54/Sigmoid_1Sigmoidlstm_cell_54/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
lstm_cell_54/mulMullstm_cell_54/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ h
lstm_cell_54/ReluRelulstm_cell_54/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_54/mul_1Mullstm_cell_54/Sigmoid:y:0lstm_cell_54/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ {
lstm_cell_54/add_1AddV2lstm_cell_54/mul:z:0lstm_cell_54/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
lstm_cell_54/Sigmoid_2Sigmoidlstm_cell_54/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ e
lstm_cell_54/Relu_1Relulstm_cell_54/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_54/mul_2Mullstm_cell_54/Sigmoid_2:y:0!lstm_cell_54/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_54_matmul_readvariableop_resource-lstm_cell_54_matmul_1_readvariableop_resource,lstm_cell_54_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_865709*
condR
while_cond_865708*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ À
NoOpNoOp$^lstm_cell_54/BiasAdd/ReadVariableOp#^lstm_cell_54/MatMul/ReadVariableOp%^lstm_cell_54/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_54/BiasAdd/ReadVariableOp#lstm_cell_54/BiasAdd/ReadVariableOp2H
"lstm_cell_54/MatMul/ReadVariableOp"lstm_cell_54/MatMul/ReadVariableOp2L
$lstm_cell_54/MatMul_1/ReadVariableOp$lstm_cell_54/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
ó
d
+__inference_salt_seq_2_layer_call_fn_866705

inputs
identity¢StatefulPartitionedCallÁ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_salt_seq_2_layer_call_and_return_conditional_losses_864309o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
£
ó
D__inference_model_12_layer_call_and_return_conditional_losses_864762
	salt_data
quantity_data!
qty_seq_864739:	!
qty_seq_864741:	 
qty_seq_864743:	"
salt_seq_864746:	"
salt_seq_864748:	 
salt_seq_864750:	"
salt_pred_864756:@
salt_pred_864758:
identity¢qty_seq/StatefulPartitionedCall¢!salt_pred/StatefulPartitionedCall¢ salt_seq/StatefulPartitionedCall
qty_seq/StatefulPartitionedCallStatefulPartitionedCallquantity_dataqty_seq_864739qty_seq_864741qty_seq_864743*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_qty_seq_layer_call_and_return_conditional_losses_864032
 salt_seq/StatefulPartitionedCallStatefulPartitionedCall	salt_datasalt_seq_864746salt_seq_864748salt_seq_864750*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_salt_seq_layer_call_and_return_conditional_losses_864182ß
salt_seq_2/PartitionedCallPartitionedCall)salt_seq/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_salt_seq_2_layer_call_and_return_conditional_losses_864195Ü
qty_seq_2/PartitionedCallPartitionedCall(qty_seq/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_qty_seq_2_layer_call_and_return_conditional_losses_864202ø
pattern/PartitionedCallPartitionedCall#salt_seq_2/PartitionedCall:output:0"qty_seq_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_pattern_layer_call_and_return_conditional_losses_864211
!salt_pred/StatefulPartitionedCallStatefulPartitionedCall pattern/PartitionedCall:output:0salt_pred_864756salt_pred_864758*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_salt_pred_layer_call_and_return_conditional_losses_864223y
IdentityIdentity*salt_pred/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯
NoOpNoOp ^qty_seq/StatefulPartitionedCall"^salt_pred/StatefulPartitionedCall!^salt_seq/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2B
qty_seq/StatefulPartitionedCallqty_seq/StatefulPartitionedCall2F
!salt_pred/StatefulPartitionedCall!salt_pred/StatefulPartitionedCall2D
 salt_seq/StatefulPartitionedCall salt_seq/StatefulPartitionedCall:V R
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	salt_data:ZV
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namequantity_data
ë
ö
-__inference_lstm_cell_55_layer_call_fn_866913

inputs
states_0
states_1
unknown:	
	unknown_0:	 
	unknown_1:	
identity

identity_1

identity_2¢StatefulPartitionedCall¨
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_55_layer_call_and_return_conditional_losses_863744o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/1
Ç

Ó
$__inference_signature_wrapper_865463
quantity_data
	salt_data
unknown:	
	unknown_0:	 
	unknown_1:	
	unknown_2:	
	unknown_3:	 
	unknown_4:	
	unknown_5:@
	unknown_6:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCall	salt_dataquantity_dataunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_863181o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namequantity_data:VR
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	salt_data
Ý

Ò
)__inference_model_12_layer_call_fn_864839
inputs_0
inputs_1
unknown:	
	unknown_0:	 
	unknown_1:	
	unknown_2:	
	unknown_3:	 
	unknown_4:	
	unknown_5:@
	unknown_6:
identity¢StatefulPartitionedCall´
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_model_12_layer_call_and_return_conditional_losses_864694o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
µ
Ã
while_cond_863947
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_863947___redundant_placeholder04
0while_while_cond_863947___redundant_placeholder14
0while_while_cond_863947___redundant_placeholder24
0while_while_cond_863947___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
8
Ð
while_body_866468
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_55_matmul_readvariableop_resource_0:	H
5while_lstm_cell_55_matmul_1_readvariableop_resource_0:	 C
4while_lstm_cell_55_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_55_matmul_readvariableop_resource:	F
3while_lstm_cell_55_matmul_1_readvariableop_resource:	 A
2while_lstm_cell_55_biasadd_readvariableop_resource:	¢)while/lstm_cell_55/BiasAdd/ReadVariableOp¢(while/lstm_cell_55/MatMul/ReadVariableOp¢*while/lstm_cell_55/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_55/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_55_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0º
while/lstm_cell_55/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_55/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
*while/lstm_cell_55/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_55_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0¡
while/lstm_cell_55/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_55/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_55/addAddV2#while/lstm_cell_55/MatMul:product:0%while/lstm_cell_55/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_55/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_55_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstm_cell_55/BiasAddBiasAddwhile/lstm_cell_55/add:z:01while/lstm_cell_55/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_55/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
while/lstm_cell_55/splitSplit+while/lstm_cell_55/split/split_dim:output:0#while/lstm_cell_55/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitz
while/lstm_cell_55/SigmoidSigmoid!while/lstm_cell_55/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
while/lstm_cell_55/Sigmoid_1Sigmoid!while/lstm_cell_55/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_55/mulMul while/lstm_cell_55/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
while/lstm_cell_55/ReluRelu!while/lstm_cell_55/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_55/mul_1Mulwhile/lstm_cell_55/Sigmoid:y:0%while/lstm_cell_55/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_55/add_1AddV2while/lstm_cell_55/mul:z:0while/lstm_cell_55/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
while/lstm_cell_55/Sigmoid_2Sigmoid!while/lstm_cell_55/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q
while/lstm_cell_55/Relu_1Reluwhile/lstm_cell_55/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_55/mul_2Mul while/lstm_cell_55/Sigmoid_2:y:0'while/lstm_cell_55/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Å
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_55/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒy
while/Identity_4Identitywhile/lstm_cell_55/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ y
while/Identity_5Identitywhile/lstm_cell_55/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ð

while/NoOpNoOp*^while/lstm_cell_55/BiasAdd/ReadVariableOp)^while/lstm_cell_55/MatMul/ReadVariableOp+^while/lstm_cell_55/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_55_biasadd_readvariableop_resource4while_lstm_cell_55_biasadd_readvariableop_resource_0"l
3while_lstm_cell_55_matmul_1_readvariableop_resource5while_lstm_cell_55_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_55_matmul_readvariableop_resource3while_lstm_cell_55_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2V
)while/lstm_cell_55/BiasAdd/ReadVariableOp)while/lstm_cell_55/BiasAdd/ReadVariableOp2T
(while/lstm_cell_55/MatMul/ReadVariableOp(while/lstm_cell_55/MatMul/ReadVariableOp2X
*while/lstm_cell_55/MatMul_1/ReadVariableOp*while/lstm_cell_55/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
8
Ð
while_body_866611
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_55_matmul_readvariableop_resource_0:	H
5while_lstm_cell_55_matmul_1_readvariableop_resource_0:	 C
4while_lstm_cell_55_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_55_matmul_readvariableop_resource:	F
3while_lstm_cell_55_matmul_1_readvariableop_resource:	 A
2while_lstm_cell_55_biasadd_readvariableop_resource:	¢)while/lstm_cell_55/BiasAdd/ReadVariableOp¢(while/lstm_cell_55/MatMul/ReadVariableOp¢*while/lstm_cell_55/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_55/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_55_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0º
while/lstm_cell_55/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_55/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
*while/lstm_cell_55/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_55_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0¡
while/lstm_cell_55/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_55/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_55/addAddV2#while/lstm_cell_55/MatMul:product:0%while/lstm_cell_55/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_55/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_55_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstm_cell_55/BiasAddBiasAddwhile/lstm_cell_55/add:z:01while/lstm_cell_55/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_55/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
while/lstm_cell_55/splitSplit+while/lstm_cell_55/split/split_dim:output:0#while/lstm_cell_55/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitz
while/lstm_cell_55/SigmoidSigmoid!while/lstm_cell_55/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
while/lstm_cell_55/Sigmoid_1Sigmoid!while/lstm_cell_55/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_55/mulMul while/lstm_cell_55/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
while/lstm_cell_55/ReluRelu!while/lstm_cell_55/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_55/mul_1Mulwhile/lstm_cell_55/Sigmoid:y:0%while/lstm_cell_55/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_55/add_1AddV2while/lstm_cell_55/mul:z:0while/lstm_cell_55/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
while/lstm_cell_55/Sigmoid_2Sigmoid!while/lstm_cell_55/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q
while/lstm_cell_55/Relu_1Reluwhile/lstm_cell_55/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_55/mul_2Mul while/lstm_cell_55/Sigmoid_2:y:0'while/lstm_cell_55/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Å
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_55/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒy
while/Identity_4Identitywhile/lstm_cell_55/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ y
while/Identity_5Identitywhile/lstm_cell_55/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ð

while/NoOpNoOp*^while/lstm_cell_55/BiasAdd/ReadVariableOp)^while/lstm_cell_55/MatMul/ReadVariableOp+^while/lstm_cell_55/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_55_biasadd_readvariableop_resource4while_lstm_cell_55_biasadd_readvariableop_resource_0"l
3while_lstm_cell_55_matmul_1_readvariableop_resource5while_lstm_cell_55_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_55_matmul_readvariableop_resource3while_lstm_cell_55_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2V
)while/lstm_cell_55/BiasAdd/ReadVariableOp)while/lstm_cell_55/BiasAdd/ReadVariableOp2T
(while/lstm_cell_55/MatMul/ReadVariableOp(while/lstm_cell_55/MatMul/ReadVariableOp2X
*while/lstm_cell_55/MatMul_1/ReadVariableOp*while/lstm_cell_55/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
Ý

H__inference_lstm_cell_55_layer_call_and_return_conditional_losses_866977

inputs
states_0
states_11
matmul_readvariableop_resource:	3
 matmul_1_readvariableop_resource:	 .
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ N
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/1
8
Ð
while_body_866182
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_55_matmul_readvariableop_resource_0:	H
5while_lstm_cell_55_matmul_1_readvariableop_resource_0:	 C
4while_lstm_cell_55_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_55_matmul_readvariableop_resource:	F
3while_lstm_cell_55_matmul_1_readvariableop_resource:	 A
2while_lstm_cell_55_biasadd_readvariableop_resource:	¢)while/lstm_cell_55/BiasAdd/ReadVariableOp¢(while/lstm_cell_55/MatMul/ReadVariableOp¢*while/lstm_cell_55/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_55/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_55_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0º
while/lstm_cell_55/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_55/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
*while/lstm_cell_55/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_55_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0¡
while/lstm_cell_55/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_55/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_55/addAddV2#while/lstm_cell_55/MatMul:product:0%while/lstm_cell_55/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_55/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_55_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstm_cell_55/BiasAddBiasAddwhile/lstm_cell_55/add:z:01while/lstm_cell_55/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_55/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
while/lstm_cell_55/splitSplit+while/lstm_cell_55/split/split_dim:output:0#while/lstm_cell_55/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitz
while/lstm_cell_55/SigmoidSigmoid!while/lstm_cell_55/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
while/lstm_cell_55/Sigmoid_1Sigmoid!while/lstm_cell_55/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_55/mulMul while/lstm_cell_55/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
while/lstm_cell_55/ReluRelu!while/lstm_cell_55/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_55/mul_1Mulwhile/lstm_cell_55/Sigmoid:y:0%while/lstm_cell_55/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_55/add_1AddV2while/lstm_cell_55/mul:z:0while/lstm_cell_55/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
while/lstm_cell_55/Sigmoid_2Sigmoid!while/lstm_cell_55/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q
while/lstm_cell_55/Relu_1Reluwhile/lstm_cell_55/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_55/mul_2Mul while/lstm_cell_55/Sigmoid_2:y:0'while/lstm_cell_55/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Å
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_55/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒy
while/Identity_4Identitywhile/lstm_cell_55/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ y
while/Identity_5Identitywhile/lstm_cell_55/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ð

while/NoOpNoOp*^while/lstm_cell_55/BiasAdd/ReadVariableOp)^while/lstm_cell_55/MatMul/ReadVariableOp+^while/lstm_cell_55/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_55_biasadd_readvariableop_resource4while_lstm_cell_55_biasadd_readvariableop_resource_0"l
3while_lstm_cell_55_matmul_1_readvariableop_resource5while_lstm_cell_55_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_55_matmul_readvariableop_resource3while_lstm_cell_55_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2V
)while/lstm_cell_55/BiasAdd/ReadVariableOp)while/lstm_cell_55/BiasAdd/ReadVariableOp2T
(while/lstm_cell_55/MatMul/ReadVariableOp(while/lstm_cell_55/MatMul/ReadVariableOp2X
*while/lstm_cell_55/MatMul_1/ReadVariableOp*while/lstm_cell_55/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 

F
*__inference_qty_seq_2_layer_call_fn_866727

inputs
identity°
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_qty_seq_2_layer_call_and_return_conditional_losses_864202`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

¼
D__inference_model_12_layer_call_and_return_conditional_losses_864789
	salt_data
quantity_data!
qty_seq_864766:	!
qty_seq_864768:	 
qty_seq_864770:	"
salt_seq_864773:	"
salt_seq_864775:	 
salt_seq_864777:	"
salt_pred_864783:@
salt_pred_864785:
identity¢qty_seq/StatefulPartitionedCall¢!qty_seq_2/StatefulPartitionedCall¢!salt_pred/StatefulPartitionedCall¢ salt_seq/StatefulPartitionedCall¢"salt_seq_2/StatefulPartitionedCall
qty_seq/StatefulPartitionedCallStatefulPartitionedCallquantity_dataqty_seq_864766qty_seq_864768qty_seq_864770*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_qty_seq_layer_call_and_return_conditional_losses_864633
 salt_seq/StatefulPartitionedCallStatefulPartitionedCall	salt_datasalt_seq_864773salt_seq_864775salt_seq_864777*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_salt_seq_layer_call_and_return_conditional_losses_864468ï
"salt_seq_2/StatefulPartitionedCallStatefulPartitionedCall)salt_seq/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_salt_seq_2_layer_call_and_return_conditional_losses_864309
!qty_seq_2/StatefulPartitionedCallStatefulPartitionedCall(qty_seq/StatefulPartitionedCall:output:0#^salt_seq_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_qty_seq_2_layer_call_and_return_conditional_losses_864286
pattern/PartitionedCallPartitionedCall+salt_seq_2/StatefulPartitionedCall:output:0*qty_seq_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_pattern_layer_call_and_return_conditional_losses_864211
!salt_pred/StatefulPartitionedCallStatefulPartitionedCall pattern/PartitionedCall:output:0salt_pred_864783salt_pred_864785*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_salt_pred_layer_call_and_return_conditional_losses_864223y
IdentityIdentity*salt_pred/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿø
NoOpNoOp ^qty_seq/StatefulPartitionedCall"^qty_seq_2/StatefulPartitionedCall"^salt_pred/StatefulPartitionedCall!^salt_seq/StatefulPartitionedCall#^salt_seq_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2B
qty_seq/StatefulPartitionedCallqty_seq/StatefulPartitionedCall2F
!qty_seq_2/StatefulPartitionedCall!qty_seq_2/StatefulPartitionedCall2F
!salt_pred/StatefulPartitionedCall!salt_pred/StatefulPartitionedCall2D
 salt_seq/StatefulPartitionedCall salt_seq/StatefulPartitionedCall2H
"salt_seq_2/StatefulPartitionedCall"salt_seq_2/StatefulPartitionedCall:V R
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	salt_data:ZV
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namequantity_data
Õ

H__inference_lstm_cell_54_layer_call_and_return_conditional_losses_863394

inputs

states
states_11
matmul_readvariableop_resource:	3
 matmul_1_readvariableop_resource:	 .
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ N
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_namestates
 
T
(__inference_pattern_layer_call_fn_866755
inputs_0
inputs_1
identity»
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_pattern_layer_call_and_return_conditional_losses_864211`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/1
8

D__inference_salt_seq_layer_call_and_return_conditional_losses_863331

inputs&
lstm_cell_54_863249:	&
lstm_cell_54_863251:	 "
lstm_cell_54_863253:	
identity¢$lstm_cell_54/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskõ
$lstm_cell_54/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_54_863249lstm_cell_54_863251lstm_cell_54_863253*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_54_layer_call_and_return_conditional_losses_863248n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ·
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_54_863249lstm_cell_54_863251lstm_cell_54_863253*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_863262*
condR
while_cond_863261*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ u
NoOpNoOp%^lstm_cell_54/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2L
$lstm_cell_54/StatefulPartitionedCall$lstm_cell_54/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÝJ

C__inference_qty_seq_layer_call_and_return_conditional_losses_866409
inputs_0>
+lstm_cell_55_matmul_readvariableop_resource:	@
-lstm_cell_55_matmul_1_readvariableop_resource:	 ;
,lstm_cell_55_biasadd_readvariableop_resource:	
identity¢#lstm_cell_55/BiasAdd/ReadVariableOp¢"lstm_cell_55/MatMul/ReadVariableOp¢$lstm_cell_55/MatMul_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_55/MatMul/ReadVariableOpReadVariableOp+lstm_cell_55_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_55/MatMulMatMulstrided_slice_2:output:0*lstm_cell_55/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_55/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_55_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0
lstm_cell_55/MatMul_1MatMulzeros:output:0,lstm_cell_55/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_55/addAddV2lstm_cell_55/MatMul:product:0lstm_cell_55/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_55/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_55_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_55/BiasAddBiasAddlstm_cell_55/add:z:0+lstm_cell_55/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_55/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstm_cell_55/splitSplit%lstm_cell_55/split/split_dim:output:0lstm_cell_55/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitn
lstm_cell_55/SigmoidSigmoidlstm_cell_55/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
lstm_cell_55/Sigmoid_1Sigmoidlstm_cell_55/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
lstm_cell_55/mulMullstm_cell_55/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ h
lstm_cell_55/ReluRelulstm_cell_55/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_55/mul_1Mullstm_cell_55/Sigmoid:y:0lstm_cell_55/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ {
lstm_cell_55/add_1AddV2lstm_cell_55/mul:z:0lstm_cell_55/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
lstm_cell_55/Sigmoid_2Sigmoidlstm_cell_55/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ e
lstm_cell_55/Relu_1Relulstm_cell_55/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_55/mul_2Mullstm_cell_55/Sigmoid_2:y:0!lstm_cell_55/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_55_matmul_readvariableop_resource-lstm_cell_55_matmul_1_readvariableop_resource,lstm_cell_55_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_866325*
condR
while_cond_866324*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ À
NoOpNoOp$^lstm_cell_55/BiasAdd/ReadVariableOp#^lstm_cell_55/MatMul/ReadVariableOp%^lstm_cell_55/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_55/BiasAdd/ReadVariableOp#lstm_cell_55/BiasAdd/ReadVariableOp2H
"lstm_cell_55/MatMul/ReadVariableOp"lstm_cell_55/MatMul/ReadVariableOp2L
$lstm_cell_55/MatMul_1/ReadVariableOp$lstm_cell_55/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
ù"
ã
while_body_863612
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_55_863636_0:	.
while_lstm_cell_55_863638_0:	 *
while_lstm_cell_55_863640_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_55_863636:	,
while_lstm_cell_55_863638:	 (
while_lstm_cell_55_863640:	¢*while/lstm_cell_55/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0³
*while/lstm_cell_55/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_55_863636_0while_lstm_cell_55_863638_0while_lstm_cell_55_863640_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_55_layer_call_and_return_conditional_losses_863598Ü
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_55/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity3while/lstm_cell_55/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/Identity_5Identity3while/lstm_cell_55/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ y

while/NoOpNoOp+^while/lstm_cell_55/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_55_863636while_lstm_cell_55_863636_0"8
while_lstm_cell_55_863638while_lstm_cell_55_863638_0"8
while_lstm_cell_55_863640while_lstm_cell_55_863640_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2X
*while/lstm_cell_55/StatefulPartitionedCall*while/lstm_cell_55/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
Õ

H__inference_lstm_cell_54_layer_call_and_return_conditional_losses_863248

inputs

states
states_11
matmul_readvariableop_resource:	3
 matmul_1_readvariableop_resource:	 .
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ N
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_namestates


e
F__inference_salt_seq_2_layer_call_and_return_conditional_losses_864309

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seed[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ÝJ

C__inference_qty_seq_layer_call_and_return_conditional_losses_866266
inputs_0>
+lstm_cell_55_matmul_readvariableop_resource:	@
-lstm_cell_55_matmul_1_readvariableop_resource:	 ;
,lstm_cell_55_biasadd_readvariableop_resource:	
identity¢#lstm_cell_55/BiasAdd/ReadVariableOp¢"lstm_cell_55/MatMul/ReadVariableOp¢$lstm_cell_55/MatMul_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_55/MatMul/ReadVariableOpReadVariableOp+lstm_cell_55_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_55/MatMulMatMulstrided_slice_2:output:0*lstm_cell_55/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_55/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_55_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0
lstm_cell_55/MatMul_1MatMulzeros:output:0,lstm_cell_55/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_55/addAddV2lstm_cell_55/MatMul:product:0lstm_cell_55/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_55/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_55_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_55/BiasAddBiasAddlstm_cell_55/add:z:0+lstm_cell_55/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_55/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstm_cell_55/splitSplit%lstm_cell_55/split/split_dim:output:0lstm_cell_55/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitn
lstm_cell_55/SigmoidSigmoidlstm_cell_55/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
lstm_cell_55/Sigmoid_1Sigmoidlstm_cell_55/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
lstm_cell_55/mulMullstm_cell_55/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ h
lstm_cell_55/ReluRelulstm_cell_55/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_55/mul_1Mullstm_cell_55/Sigmoid:y:0lstm_cell_55/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ {
lstm_cell_55/add_1AddV2lstm_cell_55/mul:z:0lstm_cell_55/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
lstm_cell_55/Sigmoid_2Sigmoidlstm_cell_55/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ e
lstm_cell_55/Relu_1Relulstm_cell_55/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_55/mul_2Mullstm_cell_55/Sigmoid_2:y:0!lstm_cell_55/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_55_matmul_readvariableop_resource-lstm_cell_55_matmul_1_readvariableop_resource,lstm_cell_55_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_866182*
condR
while_cond_866181*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ À
NoOpNoOp$^lstm_cell_55/BiasAdd/ReadVariableOp#^lstm_cell_55/MatMul/ReadVariableOp%^lstm_cell_55/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_55/BiasAdd/ReadVariableOp#lstm_cell_55/BiasAdd/ReadVariableOp2H
"lstm_cell_55/MatMul/ReadVariableOp"lstm_cell_55/MatMul/ReadVariableOp2L
$lstm_cell_55/MatMul_1/ReadVariableOp$lstm_cell_55/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
ØA
Ð

qty_seq_while_body_864899,
(qty_seq_while_qty_seq_while_loop_counter2
.qty_seq_while_qty_seq_while_maximum_iterations
qty_seq_while_placeholder
qty_seq_while_placeholder_1
qty_seq_while_placeholder_2
qty_seq_while_placeholder_3+
'qty_seq_while_qty_seq_strided_slice_1_0g
cqty_seq_while_tensorarrayv2read_tensorlistgetitem_qty_seq_tensorarrayunstack_tensorlistfromtensor_0N
;qty_seq_while_lstm_cell_55_matmul_readvariableop_resource_0:	P
=qty_seq_while_lstm_cell_55_matmul_1_readvariableop_resource_0:	 K
<qty_seq_while_lstm_cell_55_biasadd_readvariableop_resource_0:	
qty_seq_while_identity
qty_seq_while_identity_1
qty_seq_while_identity_2
qty_seq_while_identity_3
qty_seq_while_identity_4
qty_seq_while_identity_5)
%qty_seq_while_qty_seq_strided_slice_1e
aqty_seq_while_tensorarrayv2read_tensorlistgetitem_qty_seq_tensorarrayunstack_tensorlistfromtensorL
9qty_seq_while_lstm_cell_55_matmul_readvariableop_resource:	N
;qty_seq_while_lstm_cell_55_matmul_1_readvariableop_resource:	 I
:qty_seq_while_lstm_cell_55_biasadd_readvariableop_resource:	¢1qty_seq/while/lstm_cell_55/BiasAdd/ReadVariableOp¢0qty_seq/while/lstm_cell_55/MatMul/ReadVariableOp¢2qty_seq/while/lstm_cell_55/MatMul_1/ReadVariableOp
?qty_seq/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Î
1qty_seq/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemcqty_seq_while_tensorarrayv2read_tensorlistgetitem_qty_seq_tensorarrayunstack_tensorlistfromtensor_0qty_seq_while_placeholderHqty_seq/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0­
0qty_seq/while/lstm_cell_55/MatMul/ReadVariableOpReadVariableOp;qty_seq_while_lstm_cell_55_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0Ò
!qty_seq/while/lstm_cell_55/MatMulMatMul8qty_seq/while/TensorArrayV2Read/TensorListGetItem:item:08qty_seq/while/lstm_cell_55/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
2qty_seq/while/lstm_cell_55/MatMul_1/ReadVariableOpReadVariableOp=qty_seq_while_lstm_cell_55_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0¹
#qty_seq/while/lstm_cell_55/MatMul_1MatMulqty_seq_while_placeholder_2:qty_seq/while/lstm_cell_55/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
qty_seq/while/lstm_cell_55/addAddV2+qty_seq/while/lstm_cell_55/MatMul:product:0-qty_seq/while/lstm_cell_55/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
1qty_seq/while/lstm_cell_55/BiasAdd/ReadVariableOpReadVariableOp<qty_seq_while_lstm_cell_55_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0¿
"qty_seq/while/lstm_cell_55/BiasAddBiasAdd"qty_seq/while/lstm_cell_55/add:z:09qty_seq/while/lstm_cell_55/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
*qty_seq/while/lstm_cell_55/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 qty_seq/while/lstm_cell_55/splitSplit3qty_seq/while/lstm_cell_55/split/split_dim:output:0+qty_seq/while/lstm_cell_55/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split
"qty_seq/while/lstm_cell_55/SigmoidSigmoid)qty_seq/while/lstm_cell_55/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
$qty_seq/while/lstm_cell_55/Sigmoid_1Sigmoid)qty_seq/while/lstm_cell_55/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
qty_seq/while/lstm_cell_55/mulMul(qty_seq/while/lstm_cell_55/Sigmoid_1:y:0qty_seq_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
qty_seq/while/lstm_cell_55/ReluRelu)qty_seq/while/lstm_cell_55/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ °
 qty_seq/while/lstm_cell_55/mul_1Mul&qty_seq/while/lstm_cell_55/Sigmoid:y:0-qty_seq/while/lstm_cell_55/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¥
 qty_seq/while/lstm_cell_55/add_1AddV2"qty_seq/while/lstm_cell_55/mul:z:0$qty_seq/while/lstm_cell_55/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
$qty_seq/while/lstm_cell_55/Sigmoid_2Sigmoid)qty_seq/while/lstm_cell_55/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
!qty_seq/while/lstm_cell_55/Relu_1Relu$qty_seq/while/lstm_cell_55/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ´
 qty_seq/while/lstm_cell_55/mul_2Mul(qty_seq/while/lstm_cell_55/Sigmoid_2:y:0/qty_seq/while/lstm_cell_55/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ å
2qty_seq/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemqty_seq_while_placeholder_1qty_seq_while_placeholder$qty_seq/while/lstm_cell_55/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒU
qty_seq/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
qty_seq/while/addAddV2qty_seq_while_placeholderqty_seq/while/add/y:output:0*
T0*
_output_shapes
: W
qty_seq/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
qty_seq/while/add_1AddV2(qty_seq_while_qty_seq_while_loop_counterqty_seq/while/add_1/y:output:0*
T0*
_output_shapes
: q
qty_seq/while/IdentityIdentityqty_seq/while/add_1:z:0^qty_seq/while/NoOp*
T0*
_output_shapes
: 
qty_seq/while/Identity_1Identity.qty_seq_while_qty_seq_while_maximum_iterations^qty_seq/while/NoOp*
T0*
_output_shapes
: q
qty_seq/while/Identity_2Identityqty_seq/while/add:z:0^qty_seq/while/NoOp*
T0*
_output_shapes
: ±
qty_seq/while/Identity_3IdentityBqty_seq/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^qty_seq/while/NoOp*
T0*
_output_shapes
: :éèÒ
qty_seq/while/Identity_4Identity$qty_seq/while/lstm_cell_55/mul_2:z:0^qty_seq/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
qty_seq/while/Identity_5Identity$qty_seq/while/lstm_cell_55/add_1:z:0^qty_seq/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ð
qty_seq/while/NoOpNoOp2^qty_seq/while/lstm_cell_55/BiasAdd/ReadVariableOp1^qty_seq/while/lstm_cell_55/MatMul/ReadVariableOp3^qty_seq/while/lstm_cell_55/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
qty_seq_while_identityqty_seq/while/Identity:output:0"=
qty_seq_while_identity_1!qty_seq/while/Identity_1:output:0"=
qty_seq_while_identity_2!qty_seq/while/Identity_2:output:0"=
qty_seq_while_identity_3!qty_seq/while/Identity_3:output:0"=
qty_seq_while_identity_4!qty_seq/while/Identity_4:output:0"=
qty_seq_while_identity_5!qty_seq/while/Identity_5:output:0"z
:qty_seq_while_lstm_cell_55_biasadd_readvariableop_resource<qty_seq_while_lstm_cell_55_biasadd_readvariableop_resource_0"|
;qty_seq_while_lstm_cell_55_matmul_1_readvariableop_resource=qty_seq_while_lstm_cell_55_matmul_1_readvariableop_resource_0"x
9qty_seq_while_lstm_cell_55_matmul_readvariableop_resource;qty_seq_while_lstm_cell_55_matmul_readvariableop_resource_0"P
%qty_seq_while_qty_seq_strided_slice_1'qty_seq_while_qty_seq_strided_slice_1_0"È
aqty_seq_while_tensorarrayv2read_tensorlistgetitem_qty_seq_tensorarrayunstack_tensorlistfromtensorcqty_seq_while_tensorarrayv2read_tensorlistgetitem_qty_seq_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2f
1qty_seq/while/lstm_cell_55/BiasAdd/ReadVariableOp1qty_seq/while/lstm_cell_55/BiasAdd/ReadVariableOp2d
0qty_seq/while/lstm_cell_55/MatMul/ReadVariableOp0qty_seq/while/lstm_cell_55/MatMul/ReadVariableOp2h
2qty_seq/while/lstm_cell_55/MatMul_1/ReadVariableOp2qty_seq/while/lstm_cell_55/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
©J

D__inference_salt_seq_layer_call_and_return_conditional_losses_866079

inputs>
+lstm_cell_54_matmul_readvariableop_resource:	@
-lstm_cell_54_matmul_1_readvariableop_resource:	 ;
,lstm_cell_54_biasadd_readvariableop_resource:	
identity¢#lstm_cell_54/BiasAdd/ReadVariableOp¢"lstm_cell_54/MatMul/ReadVariableOp¢$lstm_cell_54/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_54/MatMul/ReadVariableOpReadVariableOp+lstm_cell_54_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_54/MatMulMatMulstrided_slice_2:output:0*lstm_cell_54/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_54/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_54_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0
lstm_cell_54/MatMul_1MatMulzeros:output:0,lstm_cell_54/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_54/addAddV2lstm_cell_54/MatMul:product:0lstm_cell_54/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_54/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_54_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_54/BiasAddBiasAddlstm_cell_54/add:z:0+lstm_cell_54/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_54/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstm_cell_54/splitSplit%lstm_cell_54/split/split_dim:output:0lstm_cell_54/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitn
lstm_cell_54/SigmoidSigmoidlstm_cell_54/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
lstm_cell_54/Sigmoid_1Sigmoidlstm_cell_54/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
lstm_cell_54/mulMullstm_cell_54/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ h
lstm_cell_54/ReluRelulstm_cell_54/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_54/mul_1Mullstm_cell_54/Sigmoid:y:0lstm_cell_54/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ {
lstm_cell_54/add_1AddV2lstm_cell_54/mul:z:0lstm_cell_54/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
lstm_cell_54/Sigmoid_2Sigmoidlstm_cell_54/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ e
lstm_cell_54/Relu_1Relulstm_cell_54/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_54/mul_2Mullstm_cell_54/Sigmoid_2:y:0!lstm_cell_54/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_54_matmul_readvariableop_resource-lstm_cell_54_matmul_1_readvariableop_resource,lstm_cell_54_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_865995*
condR
while_cond_865994*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ À
NoOpNoOp$^lstm_cell_54/BiasAdd/ReadVariableOp#^lstm_cell_54/MatMul/ReadVariableOp%^lstm_cell_54/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_54/BiasAdd/ReadVariableOp#lstm_cell_54/BiasAdd/ReadVariableOp2H
"lstm_cell_54/MatMul/ReadVariableOp"lstm_cell_54/MatMul/ReadVariableOp2L
$lstm_cell_54/MatMul_1/ReadVariableOp$lstm_cell_54/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
Ã
while_cond_864383
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_864383___redundant_placeholder04
0while_while_cond_864383___redundant_placeholder14
0while_while_cond_864383___redundant_placeholder24
0while_while_cond_864383___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
¸F

__inference__traced_save_867094
file_prefix/
+savev2_salt_pred_kernel_read_readvariableop-
)savev2_salt_pred_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop;
7savev2_salt_seq_lstm_cell_54_kernel_read_readvariableopE
Asavev2_salt_seq_lstm_cell_54_recurrent_kernel_read_readvariableop9
5savev2_salt_seq_lstm_cell_54_bias_read_readvariableop:
6savev2_qty_seq_lstm_cell_55_kernel_read_readvariableopD
@savev2_qty_seq_lstm_cell_55_recurrent_kernel_read_readvariableop8
4savev2_qty_seq_lstm_cell_55_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_salt_pred_kernel_m_read_readvariableop4
0savev2_adam_salt_pred_bias_m_read_readvariableopB
>savev2_adam_salt_seq_lstm_cell_54_kernel_m_read_readvariableopL
Hsavev2_adam_salt_seq_lstm_cell_54_recurrent_kernel_m_read_readvariableop@
<savev2_adam_salt_seq_lstm_cell_54_bias_m_read_readvariableopA
=savev2_adam_qty_seq_lstm_cell_55_kernel_m_read_readvariableopK
Gsavev2_adam_qty_seq_lstm_cell_55_recurrent_kernel_m_read_readvariableop?
;savev2_adam_qty_seq_lstm_cell_55_bias_m_read_readvariableop6
2savev2_adam_salt_pred_kernel_v_read_readvariableop4
0savev2_adam_salt_pred_bias_v_read_readvariableopB
>savev2_adam_salt_seq_lstm_cell_54_kernel_v_read_readvariableopL
Hsavev2_adam_salt_seq_lstm_cell_54_recurrent_kernel_v_read_readvariableop@
<savev2_adam_salt_seq_lstm_cell_54_bias_v_read_readvariableopA
=savev2_adam_qty_seq_lstm_cell_55_kernel_v_read_readvariableopK
Gsavev2_adam_qty_seq_lstm_cell_55_recurrent_kernel_v_read_readvariableop?
;savev2_adam_qty_seq_lstm_cell_55_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: µ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*Þ
valueÔBÑ B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH­
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_salt_pred_kernel_read_readvariableop)savev2_salt_pred_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop7savev2_salt_seq_lstm_cell_54_kernel_read_readvariableopAsavev2_salt_seq_lstm_cell_54_recurrent_kernel_read_readvariableop5savev2_salt_seq_lstm_cell_54_bias_read_readvariableop6savev2_qty_seq_lstm_cell_55_kernel_read_readvariableop@savev2_qty_seq_lstm_cell_55_recurrent_kernel_read_readvariableop4savev2_qty_seq_lstm_cell_55_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_salt_pred_kernel_m_read_readvariableop0savev2_adam_salt_pred_bias_m_read_readvariableop>savev2_adam_salt_seq_lstm_cell_54_kernel_m_read_readvariableopHsavev2_adam_salt_seq_lstm_cell_54_recurrent_kernel_m_read_readvariableop<savev2_adam_salt_seq_lstm_cell_54_bias_m_read_readvariableop=savev2_adam_qty_seq_lstm_cell_55_kernel_m_read_readvariableopGsavev2_adam_qty_seq_lstm_cell_55_recurrent_kernel_m_read_readvariableop;savev2_adam_qty_seq_lstm_cell_55_bias_m_read_readvariableop2savev2_adam_salt_pred_kernel_v_read_readvariableop0savev2_adam_salt_pred_bias_v_read_readvariableop>savev2_adam_salt_seq_lstm_cell_54_kernel_v_read_readvariableopHsavev2_adam_salt_seq_lstm_cell_54_recurrent_kernel_v_read_readvariableop<savev2_adam_salt_seq_lstm_cell_54_bias_v_read_readvariableop=savev2_adam_qty_seq_lstm_cell_55_kernel_v_read_readvariableopGsavev2_adam_qty_seq_lstm_cell_55_recurrent_kernel_v_read_readvariableop;savev2_adam_qty_seq_lstm_cell_55_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *.
dtypes$
"2 	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapesó
ð: :@:: : : : : :	:	 ::	:	 :: : :@::	:	 ::	:	 ::@::	:	 ::	:	 :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:%	!

_output_shapes
:	 :!


_output_shapes	
::%!

_output_shapes
:	:%!

_output_shapes
:	 :!

_output_shapes	
::

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@: 

_output_shapes
::%!

_output_shapes
:	:%!

_output_shapes
:	 :!

_output_shapes	
::%!

_output_shapes
:	:%!

_output_shapes
:	 :!

_output_shapes	
::$ 

_output_shapes

:@: 

_output_shapes
::%!

_output_shapes
:	:%!

_output_shapes
:	 :!

_output_shapes	
::%!

_output_shapes
:	:%!

_output_shapes
:	 :!

_output_shapes	
:: 

_output_shapes
: 
Ê
	
!__inference__wrapped_model_863181
	salt_data
quantity_dataO
<model_12_qty_seq_lstm_cell_55_matmul_readvariableop_resource:	Q
>model_12_qty_seq_lstm_cell_55_matmul_1_readvariableop_resource:	 L
=model_12_qty_seq_lstm_cell_55_biasadd_readvariableop_resource:	P
=model_12_salt_seq_lstm_cell_54_matmul_readvariableop_resource:	R
?model_12_salt_seq_lstm_cell_54_matmul_1_readvariableop_resource:	 M
>model_12_salt_seq_lstm_cell_54_biasadd_readvariableop_resource:	C
1model_12_salt_pred_matmul_readvariableop_resource:@@
2model_12_salt_pred_biasadd_readvariableop_resource:
identity¢4model_12/qty_seq/lstm_cell_55/BiasAdd/ReadVariableOp¢3model_12/qty_seq/lstm_cell_55/MatMul/ReadVariableOp¢5model_12/qty_seq/lstm_cell_55/MatMul_1/ReadVariableOp¢model_12/qty_seq/while¢)model_12/salt_pred/BiasAdd/ReadVariableOp¢(model_12/salt_pred/MatMul/ReadVariableOp¢5model_12/salt_seq/lstm_cell_54/BiasAdd/ReadVariableOp¢4model_12/salt_seq/lstm_cell_54/MatMul/ReadVariableOp¢6model_12/salt_seq/lstm_cell_54/MatMul_1/ReadVariableOp¢model_12/salt_seq/whileS
model_12/qty_seq/ShapeShapequantity_data*
T0*
_output_shapes
:n
$model_12/qty_seq/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&model_12/qty_seq/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&model_12/qty_seq/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¦
model_12/qty_seq/strided_sliceStridedSlicemodel_12/qty_seq/Shape:output:0-model_12/qty_seq/strided_slice/stack:output:0/model_12/qty_seq/strided_slice/stack_1:output:0/model_12/qty_seq/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
model_12/qty_seq/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : ¦
model_12/qty_seq/zeros/packedPack'model_12/qty_seq/strided_slice:output:0(model_12/qty_seq/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:a
model_12/qty_seq/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
model_12/qty_seq/zerosFill&model_12/qty_seq/zeros/packed:output:0%model_12/qty_seq/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
!model_12/qty_seq/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : ª
model_12/qty_seq/zeros_1/packedPack'model_12/qty_seq/strided_slice:output:0*model_12/qty_seq/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:c
model_12/qty_seq/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¥
model_12/qty_seq/zeros_1Fill(model_12/qty_seq/zeros_1/packed:output:0'model_12/qty_seq/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
model_12/qty_seq/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
model_12/qty_seq/transpose	Transposequantity_data(model_12/qty_seq/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
model_12/qty_seq/Shape_1Shapemodel_12/qty_seq/transpose:y:0*
T0*
_output_shapes
:p
&model_12/qty_seq/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(model_12/qty_seq/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(model_12/qty_seq/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 model_12/qty_seq/strided_slice_1StridedSlice!model_12/qty_seq/Shape_1:output:0/model_12/qty_seq/strided_slice_1/stack:output:01model_12/qty_seq/strided_slice_1/stack_1:output:01model_12/qty_seq/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
,model_12/qty_seq/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿç
model_12/qty_seq/TensorArrayV2TensorListReserve5model_12/qty_seq/TensorArrayV2/element_shape:output:0)model_12/qty_seq/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Fmodel_12/qty_seq/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
8model_12/qty_seq/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormodel_12/qty_seq/transpose:y:0Omodel_12/qty_seq/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒp
&model_12/qty_seq/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(model_12/qty_seq/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(model_12/qty_seq/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¾
 model_12/qty_seq/strided_slice_2StridedSlicemodel_12/qty_seq/transpose:y:0/model_12/qty_seq/strided_slice_2/stack:output:01model_12/qty_seq/strided_slice_2/stack_1:output:01model_12/qty_seq/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask±
3model_12/qty_seq/lstm_cell_55/MatMul/ReadVariableOpReadVariableOp<model_12_qty_seq_lstm_cell_55_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0É
$model_12/qty_seq/lstm_cell_55/MatMulMatMul)model_12/qty_seq/strided_slice_2:output:0;model_12/qty_seq/lstm_cell_55/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
5model_12/qty_seq/lstm_cell_55/MatMul_1/ReadVariableOpReadVariableOp>model_12_qty_seq_lstm_cell_55_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0Ã
&model_12/qty_seq/lstm_cell_55/MatMul_1MatMulmodel_12/qty_seq/zeros:output:0=model_12/qty_seq/lstm_cell_55/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
!model_12/qty_seq/lstm_cell_55/addAddV2.model_12/qty_seq/lstm_cell_55/MatMul:product:00model_12/qty_seq/lstm_cell_55/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯
4model_12/qty_seq/lstm_cell_55/BiasAdd/ReadVariableOpReadVariableOp=model_12_qty_seq_lstm_cell_55_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0È
%model_12/qty_seq/lstm_cell_55/BiasAddBiasAdd%model_12/qty_seq/lstm_cell_55/add:z:0<model_12/qty_seq/lstm_cell_55/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
-model_12/qty_seq/lstm_cell_55/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
#model_12/qty_seq/lstm_cell_55/splitSplit6model_12/qty_seq/lstm_cell_55/split/split_dim:output:0.model_12/qty_seq/lstm_cell_55/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split
%model_12/qty_seq/lstm_cell_55/SigmoidSigmoid,model_12/qty_seq/lstm_cell_55/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
'model_12/qty_seq/lstm_cell_55/Sigmoid_1Sigmoid,model_12/qty_seq/lstm_cell_55/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ª
!model_12/qty_seq/lstm_cell_55/mulMul+model_12/qty_seq/lstm_cell_55/Sigmoid_1:y:0!model_12/qty_seq/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"model_12/qty_seq/lstm_cell_55/ReluRelu,model_12/qty_seq/lstm_cell_55/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¹
#model_12/qty_seq/lstm_cell_55/mul_1Mul)model_12/qty_seq/lstm_cell_55/Sigmoid:y:00model_12/qty_seq/lstm_cell_55/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ®
#model_12/qty_seq/lstm_cell_55/add_1AddV2%model_12/qty_seq/lstm_cell_55/mul:z:0'model_12/qty_seq/lstm_cell_55/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
'model_12/qty_seq/lstm_cell_55/Sigmoid_2Sigmoid,model_12/qty_seq/lstm_cell_55/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
$model_12/qty_seq/lstm_cell_55/Relu_1Relu'model_12/qty_seq/lstm_cell_55/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ½
#model_12/qty_seq/lstm_cell_55/mul_2Mul+model_12/qty_seq/lstm_cell_55/Sigmoid_2:y:02model_12/qty_seq/lstm_cell_55/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
.model_12/qty_seq/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ë
 model_12/qty_seq/TensorArrayV2_1TensorListReserve7model_12/qty_seq/TensorArrayV2_1/element_shape:output:0)model_12/qty_seq/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒW
model_12/qty_seq/timeConst*
_output_shapes
: *
dtype0*
value	B : t
)model_12/qty_seq/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿe
#model_12/qty_seq/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ð
model_12/qty_seq/whileWhile,model_12/qty_seq/while/loop_counter:output:02model_12/qty_seq/while/maximum_iterations:output:0model_12/qty_seq/time:output:0)model_12/qty_seq/TensorArrayV2_1:handle:0model_12/qty_seq/zeros:output:0!model_12/qty_seq/zeros_1:output:0)model_12/qty_seq/strided_slice_1:output:0Hmodel_12/qty_seq/TensorArrayUnstack/TensorListFromTensor:output_handle:0<model_12_qty_seq_lstm_cell_55_matmul_readvariableop_resource>model_12_qty_seq_lstm_cell_55_matmul_1_readvariableop_resource=model_12_qty_seq_lstm_cell_55_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *.
body&R$
"model_12_qty_seq_while_body_862948*.
cond&R$
"model_12_qty_seq_while_cond_862947*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
Amodel_12/qty_seq/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    õ
3model_12/qty_seq/TensorArrayV2Stack/TensorListStackTensorListStackmodel_12/qty_seq/while:output:3Jmodel_12/qty_seq/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0y
&model_12/qty_seq/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿr
(model_12/qty_seq/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: r
(model_12/qty_seq/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ü
 model_12/qty_seq/strided_slice_3StridedSlice<model_12/qty_seq/TensorArrayV2Stack/TensorListStack:tensor:0/model_12/qty_seq/strided_slice_3/stack:output:01model_12/qty_seq/strided_slice_3/stack_1:output:01model_12/qty_seq/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maskv
!model_12/qty_seq/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          É
model_12/qty_seq/transpose_1	Transpose<model_12/qty_seq/TensorArrayV2Stack/TensorListStack:tensor:0*model_12/qty_seq/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ l
model_12/qty_seq/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    P
model_12/salt_seq/ShapeShape	salt_data*
T0*
_output_shapes
:o
%model_12/salt_seq/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'model_12/salt_seq/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'model_12/salt_seq/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:«
model_12/salt_seq/strided_sliceStridedSlice model_12/salt_seq/Shape:output:0.model_12/salt_seq/strided_slice/stack:output:00model_12/salt_seq/strided_slice/stack_1:output:00model_12/salt_seq/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 model_12/salt_seq/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : ©
model_12/salt_seq/zeros/packedPack(model_12/salt_seq/strided_slice:output:0)model_12/salt_seq/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:b
model_12/salt_seq/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¢
model_12/salt_seq/zerosFill'model_12/salt_seq/zeros/packed:output:0&model_12/salt_seq/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ d
"model_12/salt_seq/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : ­
 model_12/salt_seq/zeros_1/packedPack(model_12/salt_seq/strided_slice:output:0+model_12/salt_seq/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:d
model_12/salt_seq/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¨
model_12/salt_seq/zeros_1Fill)model_12/salt_seq/zeros_1/packed:output:0(model_12/salt_seq/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ u
 model_12/salt_seq/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
model_12/salt_seq/transpose	Transpose	salt_data)model_12/salt_seq/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
model_12/salt_seq/Shape_1Shapemodel_12/salt_seq/transpose:y:0*
T0*
_output_shapes
:q
'model_12/salt_seq/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)model_12/salt_seq/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)model_12/salt_seq/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
!model_12/salt_seq/strided_slice_1StridedSlice"model_12/salt_seq/Shape_1:output:00model_12/salt_seq/strided_slice_1/stack:output:02model_12/salt_seq/strided_slice_1/stack_1:output:02model_12/salt_seq/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
-model_12/salt_seq/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿê
model_12/salt_seq/TensorArrayV2TensorListReserve6model_12/salt_seq/TensorArrayV2/element_shape:output:0*model_12/salt_seq/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Gmodel_12/salt_seq/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
9model_12/salt_seq/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormodel_12/salt_seq/transpose:y:0Pmodel_12/salt_seq/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒq
'model_12/salt_seq/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)model_12/salt_seq/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)model_12/salt_seq/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ã
!model_12/salt_seq/strided_slice_2StridedSlicemodel_12/salt_seq/transpose:y:00model_12/salt_seq/strided_slice_2/stack:output:02model_12/salt_seq/strided_slice_2/stack_1:output:02model_12/salt_seq/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask³
4model_12/salt_seq/lstm_cell_54/MatMul/ReadVariableOpReadVariableOp=model_12_salt_seq_lstm_cell_54_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0Ì
%model_12/salt_seq/lstm_cell_54/MatMulMatMul*model_12/salt_seq/strided_slice_2:output:0<model_12/salt_seq/lstm_cell_54/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
6model_12/salt_seq/lstm_cell_54/MatMul_1/ReadVariableOpReadVariableOp?model_12_salt_seq_lstm_cell_54_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0Æ
'model_12/salt_seq/lstm_cell_54/MatMul_1MatMul model_12/salt_seq/zeros:output:0>model_12/salt_seq/lstm_cell_54/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
"model_12/salt_seq/lstm_cell_54/addAddV2/model_12/salt_seq/lstm_cell_54/MatMul:product:01model_12/salt_seq/lstm_cell_54/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
5model_12/salt_seq/lstm_cell_54/BiasAdd/ReadVariableOpReadVariableOp>model_12_salt_seq_lstm_cell_54_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ë
&model_12/salt_seq/lstm_cell_54/BiasAddBiasAdd&model_12/salt_seq/lstm_cell_54/add:z:0=model_12/salt_seq/lstm_cell_54/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
.model_12/salt_seq/lstm_cell_54/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
$model_12/salt_seq/lstm_cell_54/splitSplit7model_12/salt_seq/lstm_cell_54/split/split_dim:output:0/model_12/salt_seq/lstm_cell_54/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split
&model_12/salt_seq/lstm_cell_54/SigmoidSigmoid-model_12/salt_seq/lstm_cell_54/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(model_12/salt_seq/lstm_cell_54/Sigmoid_1Sigmoid-model_12/salt_seq/lstm_cell_54/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ­
"model_12/salt_seq/lstm_cell_54/mulMul,model_12/salt_seq/lstm_cell_54/Sigmoid_1:y:0"model_12/salt_seq/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
#model_12/salt_seq/lstm_cell_54/ReluRelu-model_12/salt_seq/lstm_cell_54/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¼
$model_12/salt_seq/lstm_cell_54/mul_1Mul*model_12/salt_seq/lstm_cell_54/Sigmoid:y:01model_12/salt_seq/lstm_cell_54/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ±
$model_12/salt_seq/lstm_cell_54/add_1AddV2&model_12/salt_seq/lstm_cell_54/mul:z:0(model_12/salt_seq/lstm_cell_54/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(model_12/salt_seq/lstm_cell_54/Sigmoid_2Sigmoid-model_12/salt_seq/lstm_cell_54/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
%model_12/salt_seq/lstm_cell_54/Relu_1Relu(model_12/salt_seq/lstm_cell_54/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ À
$model_12/salt_seq/lstm_cell_54/mul_2Mul,model_12/salt_seq/lstm_cell_54/Sigmoid_2:y:03model_12/salt_seq/lstm_cell_54/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
/model_12/salt_seq/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    î
!model_12/salt_seq/TensorArrayV2_1TensorListReserve8model_12/salt_seq/TensorArrayV2_1/element_shape:output:0*model_12/salt_seq/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒX
model_12/salt_seq/timeConst*
_output_shapes
: *
dtype0*
value	B : u
*model_12/salt_seq/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿf
$model_12/salt_seq/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : þ
model_12/salt_seq/whileWhile-model_12/salt_seq/while/loop_counter:output:03model_12/salt_seq/while/maximum_iterations:output:0model_12/salt_seq/time:output:0*model_12/salt_seq/TensorArrayV2_1:handle:0 model_12/salt_seq/zeros:output:0"model_12/salt_seq/zeros_1:output:0*model_12/salt_seq/strided_slice_1:output:0Imodel_12/salt_seq/TensorArrayUnstack/TensorListFromTensor:output_handle:0=model_12_salt_seq_lstm_cell_54_matmul_readvariableop_resource?model_12_salt_seq_lstm_cell_54_matmul_1_readvariableop_resource>model_12_salt_seq_lstm_cell_54_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( */
body'R%
#model_12_salt_seq_while_body_863087*/
cond'R%
#model_12_salt_seq_while_cond_863086*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
Bmodel_12/salt_seq/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ø
4model_12/salt_seq/TensorArrayV2Stack/TensorListStackTensorListStack model_12/salt_seq/while:output:3Kmodel_12/salt_seq/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0z
'model_12/salt_seq/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿs
)model_12/salt_seq/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: s
)model_12/salt_seq/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:á
!model_12/salt_seq/strided_slice_3StridedSlice=model_12/salt_seq/TensorArrayV2Stack/TensorListStack:tensor:00model_12/salt_seq/strided_slice_3/stack:output:02model_12/salt_seq/strided_slice_3/stack_1:output:02model_12/salt_seq/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maskw
"model_12/salt_seq/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ì
model_12/salt_seq/transpose_1	Transpose=model_12/salt_seq/TensorArrayV2Stack/TensorListStack:tensor:0+model_12/salt_seq/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ m
model_12/salt_seq/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
model_12/salt_seq_2/IdentityIdentity*model_12/salt_seq/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
model_12/qty_seq_2/IdentityIdentity)model_12/qty_seq/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ^
model_12/pattern/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ò
model_12/pattern/concatConcatV2%model_12/salt_seq_2/Identity:output:0$model_12/qty_seq_2/Identity:output:0%model_12/pattern/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(model_12/salt_pred/MatMul/ReadVariableOpReadVariableOp1model_12_salt_pred_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0©
model_12/salt_pred/MatMulMatMul model_12/pattern/concat:output:00model_12/salt_pred/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)model_12/salt_pred/BiasAdd/ReadVariableOpReadVariableOp2model_12_salt_pred_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¯
model_12/salt_pred/BiasAddBiasAdd#model_12/salt_pred/MatMul:product:01model_12/salt_pred/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
IdentityIdentity#model_12/salt_pred/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp5^model_12/qty_seq/lstm_cell_55/BiasAdd/ReadVariableOp4^model_12/qty_seq/lstm_cell_55/MatMul/ReadVariableOp6^model_12/qty_seq/lstm_cell_55/MatMul_1/ReadVariableOp^model_12/qty_seq/while*^model_12/salt_pred/BiasAdd/ReadVariableOp)^model_12/salt_pred/MatMul/ReadVariableOp6^model_12/salt_seq/lstm_cell_54/BiasAdd/ReadVariableOp5^model_12/salt_seq/lstm_cell_54/MatMul/ReadVariableOp7^model_12/salt_seq/lstm_cell_54/MatMul_1/ReadVariableOp^model_12/salt_seq/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2l
4model_12/qty_seq/lstm_cell_55/BiasAdd/ReadVariableOp4model_12/qty_seq/lstm_cell_55/BiasAdd/ReadVariableOp2j
3model_12/qty_seq/lstm_cell_55/MatMul/ReadVariableOp3model_12/qty_seq/lstm_cell_55/MatMul/ReadVariableOp2n
5model_12/qty_seq/lstm_cell_55/MatMul_1/ReadVariableOp5model_12/qty_seq/lstm_cell_55/MatMul_1/ReadVariableOp20
model_12/qty_seq/whilemodel_12/qty_seq/while2V
)model_12/salt_pred/BiasAdd/ReadVariableOp)model_12/salt_pred/BiasAdd/ReadVariableOp2T
(model_12/salt_pred/MatMul/ReadVariableOp(model_12/salt_pred/MatMul/ReadVariableOp2n
5model_12/salt_seq/lstm_cell_54/BiasAdd/ReadVariableOp5model_12/salt_seq/lstm_cell_54/BiasAdd/ReadVariableOp2l
4model_12/salt_seq/lstm_cell_54/MatMul/ReadVariableOp4model_12/salt_seq/lstm_cell_54/MatMul/ReadVariableOp2p
6model_12/salt_seq/lstm_cell_54/MatMul_1/ReadVariableOp6model_12/salt_seq/lstm_cell_54/MatMul_1/ReadVariableOp22
model_12/salt_seq/whilemodel_12/salt_seq/while:V R
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	salt_data:ZV
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namequantity_data
÷
µ
(__inference_qty_seq_layer_call_fn_866112

inputs
unknown:	
	unknown_0:	 
	unknown_1:	
identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_qty_seq_layer_call_and_return_conditional_losses_864032o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È	
ö
E__inference_salt_pred_layer_call_and_return_conditional_losses_864223

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ë
ö
-__inference_lstm_cell_55_layer_call_fn_866896

inputs
states_0
states_1
unknown:	
	unknown_0:	 
	unknown_1:	
identity

identity_1

identity_2¢StatefulPartitionedCall¨
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_55_layer_call_and_return_conditional_losses_863598o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/1
÷
µ
(__inference_qty_seq_layer_call_fn_866123

inputs
unknown:	
	unknown_0:	 
	unknown_1:	
identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_qty_seq_layer_call_and_return_conditional_losses_864633o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²

÷
salt_seq_while_cond_865330.
*salt_seq_while_salt_seq_while_loop_counter4
0salt_seq_while_salt_seq_while_maximum_iterations
salt_seq_while_placeholder 
salt_seq_while_placeholder_1 
salt_seq_while_placeholder_2 
salt_seq_while_placeholder_30
,salt_seq_while_less_salt_seq_strided_slice_1F
Bsalt_seq_while_salt_seq_while_cond_865330___redundant_placeholder0F
Bsalt_seq_while_salt_seq_while_cond_865330___redundant_placeholder1F
Bsalt_seq_while_salt_seq_while_cond_865330___redundant_placeholder2F
Bsalt_seq_while_salt_seq_while_cond_865330___redundant_placeholder3
salt_seq_while_identity

salt_seq/while/LessLesssalt_seq_while_placeholder,salt_seq_while_less_salt_seq_strided_slice_1*
T0*
_output_shapes
: ]
salt_seq/while/IdentityIdentitysalt_seq/while/Less:z:0*
T0
*
_output_shapes
: ";
salt_seq_while_identity salt_seq/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
µ
Ã
while_cond_864548
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_864548___redundant_placeholder04
0while_while_cond_864548___redundant_placeholder14
0while_while_cond_864548___redundant_placeholder24
0while_while_cond_864548___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
ÞJ

D__inference_salt_seq_layer_call_and_return_conditional_losses_865650
inputs_0>
+lstm_cell_54_matmul_readvariableop_resource:	@
-lstm_cell_54_matmul_1_readvariableop_resource:	 ;
,lstm_cell_54_biasadd_readvariableop_resource:	
identity¢#lstm_cell_54/BiasAdd/ReadVariableOp¢"lstm_cell_54/MatMul/ReadVariableOp¢$lstm_cell_54/MatMul_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_54/MatMul/ReadVariableOpReadVariableOp+lstm_cell_54_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_54/MatMulMatMulstrided_slice_2:output:0*lstm_cell_54/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_54/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_54_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0
lstm_cell_54/MatMul_1MatMulzeros:output:0,lstm_cell_54/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_54/addAddV2lstm_cell_54/MatMul:product:0lstm_cell_54/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_54/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_54_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_54/BiasAddBiasAddlstm_cell_54/add:z:0+lstm_cell_54/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_54/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstm_cell_54/splitSplit%lstm_cell_54/split/split_dim:output:0lstm_cell_54/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitn
lstm_cell_54/SigmoidSigmoidlstm_cell_54/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
lstm_cell_54/Sigmoid_1Sigmoidlstm_cell_54/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
lstm_cell_54/mulMullstm_cell_54/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ h
lstm_cell_54/ReluRelulstm_cell_54/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_54/mul_1Mullstm_cell_54/Sigmoid:y:0lstm_cell_54/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ {
lstm_cell_54/add_1AddV2lstm_cell_54/mul:z:0lstm_cell_54/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
lstm_cell_54/Sigmoid_2Sigmoidlstm_cell_54/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ e
lstm_cell_54/Relu_1Relulstm_cell_54/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_54/mul_2Mullstm_cell_54/Sigmoid_2:y:0!lstm_cell_54/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_54_matmul_readvariableop_resource-lstm_cell_54_matmul_1_readvariableop_resource,lstm_cell_54_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_865566*
condR
while_cond_865565*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ À
NoOpNoOp$^lstm_cell_54/BiasAdd/ReadVariableOp#^lstm_cell_54/MatMul/ReadVariableOp%^lstm_cell_54/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_54/BiasAdd/ReadVariableOp#lstm_cell_54/BiasAdd/ReadVariableOp2H
"lstm_cell_54/MatMul/ReadVariableOp"lstm_cell_54/MatMul/ReadVariableOp2L
$lstm_cell_54/MatMul_1/ReadVariableOp$lstm_cell_54/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
¨J

C__inference_qty_seq_layer_call_and_return_conditional_losses_866552

inputs>
+lstm_cell_55_matmul_readvariableop_resource:	@
-lstm_cell_55_matmul_1_readvariableop_resource:	 ;
,lstm_cell_55_biasadd_readvariableop_resource:	
identity¢#lstm_cell_55/BiasAdd/ReadVariableOp¢"lstm_cell_55/MatMul/ReadVariableOp¢$lstm_cell_55/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_55/MatMul/ReadVariableOpReadVariableOp+lstm_cell_55_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_55/MatMulMatMulstrided_slice_2:output:0*lstm_cell_55/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_55/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_55_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0
lstm_cell_55/MatMul_1MatMulzeros:output:0,lstm_cell_55/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_55/addAddV2lstm_cell_55/MatMul:product:0lstm_cell_55/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_55/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_55_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_55/BiasAddBiasAddlstm_cell_55/add:z:0+lstm_cell_55/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_55/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstm_cell_55/splitSplit%lstm_cell_55/split/split_dim:output:0lstm_cell_55/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitn
lstm_cell_55/SigmoidSigmoidlstm_cell_55/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
lstm_cell_55/Sigmoid_1Sigmoidlstm_cell_55/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
lstm_cell_55/mulMullstm_cell_55/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ h
lstm_cell_55/ReluRelulstm_cell_55/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_55/mul_1Mullstm_cell_55/Sigmoid:y:0lstm_cell_55/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ {
lstm_cell_55/add_1AddV2lstm_cell_55/mul:z:0lstm_cell_55/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
lstm_cell_55/Sigmoid_2Sigmoidlstm_cell_55/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ e
lstm_cell_55/Relu_1Relulstm_cell_55/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_55/mul_2Mullstm_cell_55/Sigmoid_2:y:0!lstm_cell_55/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_55_matmul_readvariableop_resource-lstm_cell_55_matmul_1_readvariableop_resource,lstm_cell_55_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_866468*
condR
while_cond_866467*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ À
NoOpNoOp$^lstm_cell_55/BiasAdd/ReadVariableOp#^lstm_cell_55/MatMul/ReadVariableOp%^lstm_cell_55/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_55/BiasAdd/ReadVariableOp#lstm_cell_55/BiasAdd/ReadVariableOp2H
"lstm_cell_55/MatMul/ReadVariableOp"lstm_cell_55/MatMul/ReadVariableOp2L
$lstm_cell_55/MatMul_1/ReadVariableOp$lstm_cell_55/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨J

C__inference_qty_seq_layer_call_and_return_conditional_losses_864032

inputs>
+lstm_cell_55_matmul_readvariableop_resource:	@
-lstm_cell_55_matmul_1_readvariableop_resource:	 ;
,lstm_cell_55_biasadd_readvariableop_resource:	
identity¢#lstm_cell_55/BiasAdd/ReadVariableOp¢"lstm_cell_55/MatMul/ReadVariableOp¢$lstm_cell_55/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_55/MatMul/ReadVariableOpReadVariableOp+lstm_cell_55_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_55/MatMulMatMulstrided_slice_2:output:0*lstm_cell_55/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_55/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_55_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0
lstm_cell_55/MatMul_1MatMulzeros:output:0,lstm_cell_55/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_55/addAddV2lstm_cell_55/MatMul:product:0lstm_cell_55/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_55/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_55_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_55/BiasAddBiasAddlstm_cell_55/add:z:0+lstm_cell_55/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_55/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstm_cell_55/splitSplit%lstm_cell_55/split/split_dim:output:0lstm_cell_55/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitn
lstm_cell_55/SigmoidSigmoidlstm_cell_55/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
lstm_cell_55/Sigmoid_1Sigmoidlstm_cell_55/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
lstm_cell_55/mulMullstm_cell_55/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ h
lstm_cell_55/ReluRelulstm_cell_55/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_55/mul_1Mullstm_cell_55/Sigmoid:y:0lstm_cell_55/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ {
lstm_cell_55/add_1AddV2lstm_cell_55/mul:z:0lstm_cell_55/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
lstm_cell_55/Sigmoid_2Sigmoidlstm_cell_55/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ e
lstm_cell_55/Relu_1Relulstm_cell_55/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_55/mul_2Mullstm_cell_55/Sigmoid_2:y:0!lstm_cell_55/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_55_matmul_readvariableop_resource-lstm_cell_55_matmul_1_readvariableop_resource,lstm_cell_55_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_863948*
condR
while_cond_863947*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ À
NoOpNoOp$^lstm_cell_55/BiasAdd/ReadVariableOp#^lstm_cell_55/MatMul/ReadVariableOp%^lstm_cell_55/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_55/BiasAdd/ReadVariableOp#lstm_cell_55/BiasAdd/ReadVariableOp2H
"lstm_cell_55/MatMul/ReadVariableOp"lstm_cell_55/MatMul/ReadVariableOp2L
$lstm_cell_55/MatMul_1/ReadVariableOp$lstm_cell_55/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ý

Ò
)__inference_model_12_layer_call_fn_864817
inputs_0
inputs_1
unknown:	
	unknown_0:	 
	unknown_1:	
	unknown_2:	
	unknown_3:	 
	unknown_4:	
	unknown_5:@
	unknown_6:
identity¢StatefulPartitionedCall´
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_model_12_layer_call_and_return_conditional_losses_864230o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
Ù
d
F__inference_salt_seq_2_layer_call_and_return_conditional_losses_866710

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ë
ö
-__inference_lstm_cell_54_layer_call_fn_866815

inputs
states_0
states_1
unknown:	
	unknown_0:	 
	unknown_1:	
identity

identity_1

identity_2¢StatefulPartitionedCall¨
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_54_layer_call_and_return_conditional_losses_863394o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/1
8

D__inference_salt_seq_layer_call_and_return_conditional_losses_863522

inputs&
lstm_cell_54_863440:	&
lstm_cell_54_863442:	 "
lstm_cell_54_863444:	
identity¢$lstm_cell_54/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskõ
$lstm_cell_54/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_54_863440lstm_cell_54_863442lstm_cell_54_863444*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_54_layer_call_and_return_conditional_losses_863394n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ·
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_54_863440lstm_cell_54_863442lstm_cell_54_863444*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_863453*
condR
while_cond_863452*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ u
NoOpNoOp%^lstm_cell_54/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2L
$lstm_cell_54/StatefulPartitionedCall$lstm_cell_54/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
8
Ð
while_body_864549
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_55_matmul_readvariableop_resource_0:	H
5while_lstm_cell_55_matmul_1_readvariableop_resource_0:	 C
4while_lstm_cell_55_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_55_matmul_readvariableop_resource:	F
3while_lstm_cell_55_matmul_1_readvariableop_resource:	 A
2while_lstm_cell_55_biasadd_readvariableop_resource:	¢)while/lstm_cell_55/BiasAdd/ReadVariableOp¢(while/lstm_cell_55/MatMul/ReadVariableOp¢*while/lstm_cell_55/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_55/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_55_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0º
while/lstm_cell_55/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_55/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
*while/lstm_cell_55/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_55_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0¡
while/lstm_cell_55/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_55/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_55/addAddV2#while/lstm_cell_55/MatMul:product:0%while/lstm_cell_55/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_55/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_55_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstm_cell_55/BiasAddBiasAddwhile/lstm_cell_55/add:z:01while/lstm_cell_55/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_55/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
while/lstm_cell_55/splitSplit+while/lstm_cell_55/split/split_dim:output:0#while/lstm_cell_55/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitz
while/lstm_cell_55/SigmoidSigmoid!while/lstm_cell_55/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
while/lstm_cell_55/Sigmoid_1Sigmoid!while/lstm_cell_55/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_55/mulMul while/lstm_cell_55/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
while/lstm_cell_55/ReluRelu!while/lstm_cell_55/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_55/mul_1Mulwhile/lstm_cell_55/Sigmoid:y:0%while/lstm_cell_55/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_55/add_1AddV2while/lstm_cell_55/mul:z:0while/lstm_cell_55/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
while/lstm_cell_55/Sigmoid_2Sigmoid!while/lstm_cell_55/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q
while/lstm_cell_55/Relu_1Reluwhile/lstm_cell_55/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_55/mul_2Mul while/lstm_cell_55/Sigmoid_2:y:0'while/lstm_cell_55/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Å
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_55/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒy
while/Identity_4Identitywhile/lstm_cell_55/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ y
while/Identity_5Identitywhile/lstm_cell_55/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ð

while/NoOpNoOp*^while/lstm_cell_55/BiasAdd/ReadVariableOp)^while/lstm_cell_55/MatMul/ReadVariableOp+^while/lstm_cell_55/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_55_biasadd_readvariableop_resource4while_lstm_cell_55_biasadd_readvariableop_resource_0"l
3while_lstm_cell_55_matmul_1_readvariableop_resource5while_lstm_cell_55_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_55_matmul_readvariableop_resource3while_lstm_cell_55_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2V
)while/lstm_cell_55/BiasAdd/ReadVariableOp)while/lstm_cell_55/BiasAdd/ReadVariableOp2T
(while/lstm_cell_55/MatMul/ReadVariableOp(while/lstm_cell_55/MatMul/ReadVariableOp2X
*while/lstm_cell_55/MatMul_1/ReadVariableOp*while/lstm_cell_55/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
îB
ð

salt_seq_while_body_865331.
*salt_seq_while_salt_seq_while_loop_counter4
0salt_seq_while_salt_seq_while_maximum_iterations
salt_seq_while_placeholder 
salt_seq_while_placeholder_1 
salt_seq_while_placeholder_2 
salt_seq_while_placeholder_3-
)salt_seq_while_salt_seq_strided_slice_1_0i
esalt_seq_while_tensorarrayv2read_tensorlistgetitem_salt_seq_tensorarrayunstack_tensorlistfromtensor_0O
<salt_seq_while_lstm_cell_54_matmul_readvariableop_resource_0:	Q
>salt_seq_while_lstm_cell_54_matmul_1_readvariableop_resource_0:	 L
=salt_seq_while_lstm_cell_54_biasadd_readvariableop_resource_0:	
salt_seq_while_identity
salt_seq_while_identity_1
salt_seq_while_identity_2
salt_seq_while_identity_3
salt_seq_while_identity_4
salt_seq_while_identity_5+
'salt_seq_while_salt_seq_strided_slice_1g
csalt_seq_while_tensorarrayv2read_tensorlistgetitem_salt_seq_tensorarrayunstack_tensorlistfromtensorM
:salt_seq_while_lstm_cell_54_matmul_readvariableop_resource:	O
<salt_seq_while_lstm_cell_54_matmul_1_readvariableop_resource:	 J
;salt_seq_while_lstm_cell_54_biasadd_readvariableop_resource:	¢2salt_seq/while/lstm_cell_54/BiasAdd/ReadVariableOp¢1salt_seq/while/lstm_cell_54/MatMul/ReadVariableOp¢3salt_seq/while/lstm_cell_54/MatMul_1/ReadVariableOp
@salt_seq/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ó
2salt_seq/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemesalt_seq_while_tensorarrayv2read_tensorlistgetitem_salt_seq_tensorarrayunstack_tensorlistfromtensor_0salt_seq_while_placeholderIsalt_seq/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0¯
1salt_seq/while/lstm_cell_54/MatMul/ReadVariableOpReadVariableOp<salt_seq_while_lstm_cell_54_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0Õ
"salt_seq/while/lstm_cell_54/MatMulMatMul9salt_seq/while/TensorArrayV2Read/TensorListGetItem:item:09salt_seq/while/lstm_cell_54/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
3salt_seq/while/lstm_cell_54/MatMul_1/ReadVariableOpReadVariableOp>salt_seq_while_lstm_cell_54_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0¼
$salt_seq/while/lstm_cell_54/MatMul_1MatMulsalt_seq_while_placeholder_2;salt_seq/while/lstm_cell_54/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
salt_seq/while/lstm_cell_54/addAddV2,salt_seq/while/lstm_cell_54/MatMul:product:0.salt_seq/while/lstm_cell_54/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
2salt_seq/while/lstm_cell_54/BiasAdd/ReadVariableOpReadVariableOp=salt_seq_while_lstm_cell_54_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0Â
#salt_seq/while/lstm_cell_54/BiasAddBiasAdd#salt_seq/while/lstm_cell_54/add:z:0:salt_seq/while/lstm_cell_54/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
+salt_seq/while/lstm_cell_54/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
!salt_seq/while/lstm_cell_54/splitSplit4salt_seq/while/lstm_cell_54/split/split_dim:output:0,salt_seq/while/lstm_cell_54/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split
#salt_seq/while/lstm_cell_54/SigmoidSigmoid*salt_seq/while/lstm_cell_54/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
%salt_seq/while/lstm_cell_54/Sigmoid_1Sigmoid*salt_seq/while/lstm_cell_54/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¡
salt_seq/while/lstm_cell_54/mulMul)salt_seq/while/lstm_cell_54/Sigmoid_1:y:0salt_seq_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 salt_seq/while/lstm_cell_54/ReluRelu*salt_seq/while/lstm_cell_54/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ³
!salt_seq/while/lstm_cell_54/mul_1Mul'salt_seq/while/lstm_cell_54/Sigmoid:y:0.salt_seq/while/lstm_cell_54/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¨
!salt_seq/while/lstm_cell_54/add_1AddV2#salt_seq/while/lstm_cell_54/mul:z:0%salt_seq/while/lstm_cell_54/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
%salt_seq/while/lstm_cell_54/Sigmoid_2Sigmoid*salt_seq/while/lstm_cell_54/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"salt_seq/while/lstm_cell_54/Relu_1Relu%salt_seq/while/lstm_cell_54/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ·
!salt_seq/while/lstm_cell_54/mul_2Mul)salt_seq/while/lstm_cell_54/Sigmoid_2:y:00salt_seq/while/lstm_cell_54/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ é
3salt_seq/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemsalt_seq_while_placeholder_1salt_seq_while_placeholder%salt_seq/while/lstm_cell_54/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒV
salt_seq/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :w
salt_seq/while/addAddV2salt_seq_while_placeholdersalt_seq/while/add/y:output:0*
T0*
_output_shapes
: X
salt_seq/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
salt_seq/while/add_1AddV2*salt_seq_while_salt_seq_while_loop_countersalt_seq/while/add_1/y:output:0*
T0*
_output_shapes
: t
salt_seq/while/IdentityIdentitysalt_seq/while/add_1:z:0^salt_seq/while/NoOp*
T0*
_output_shapes
: 
salt_seq/while/Identity_1Identity0salt_seq_while_salt_seq_while_maximum_iterations^salt_seq/while/NoOp*
T0*
_output_shapes
: t
salt_seq/while/Identity_2Identitysalt_seq/while/add:z:0^salt_seq/while/NoOp*
T0*
_output_shapes
: ´
salt_seq/while/Identity_3IdentityCsalt_seq/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^salt_seq/while/NoOp*
T0*
_output_shapes
: :éèÒ
salt_seq/while/Identity_4Identity%salt_seq/while/lstm_cell_54/mul_2:z:0^salt_seq/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
salt_seq/while/Identity_5Identity%salt_seq/while/lstm_cell_54/add_1:z:0^salt_seq/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ô
salt_seq/while/NoOpNoOp3^salt_seq/while/lstm_cell_54/BiasAdd/ReadVariableOp2^salt_seq/while/lstm_cell_54/MatMul/ReadVariableOp4^salt_seq/while/lstm_cell_54/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ";
salt_seq_while_identity salt_seq/while/Identity:output:0"?
salt_seq_while_identity_1"salt_seq/while/Identity_1:output:0"?
salt_seq_while_identity_2"salt_seq/while/Identity_2:output:0"?
salt_seq_while_identity_3"salt_seq/while/Identity_3:output:0"?
salt_seq_while_identity_4"salt_seq/while/Identity_4:output:0"?
salt_seq_while_identity_5"salt_seq/while/Identity_5:output:0"|
;salt_seq_while_lstm_cell_54_biasadd_readvariableop_resource=salt_seq_while_lstm_cell_54_biasadd_readvariableop_resource_0"~
<salt_seq_while_lstm_cell_54_matmul_1_readvariableop_resource>salt_seq_while_lstm_cell_54_matmul_1_readvariableop_resource_0"z
:salt_seq_while_lstm_cell_54_matmul_readvariableop_resource<salt_seq_while_lstm_cell_54_matmul_readvariableop_resource_0"T
'salt_seq_while_salt_seq_strided_slice_1)salt_seq_while_salt_seq_strided_slice_1_0"Ì
csalt_seq_while_tensorarrayv2read_tensorlistgetitem_salt_seq_tensorarrayunstack_tensorlistfromtensoresalt_seq_while_tensorarrayv2read_tensorlistgetitem_salt_seq_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2h
2salt_seq/while/lstm_cell_54/BiasAdd/ReadVariableOp2salt_seq/while/lstm_cell_54/BiasAdd/ReadVariableOp2f
1salt_seq/while/lstm_cell_54/MatMul/ReadVariableOp1salt_seq/while/lstm_cell_54/MatMul/ReadVariableOp2j
3salt_seq/while/lstm_cell_54/MatMul_1/ReadVariableOp3salt_seq/while/lstm_cell_54/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
©J

D__inference_salt_seq_layer_call_and_return_conditional_losses_864182

inputs>
+lstm_cell_54_matmul_readvariableop_resource:	@
-lstm_cell_54_matmul_1_readvariableop_resource:	 ;
,lstm_cell_54_biasadd_readvariableop_resource:	
identity¢#lstm_cell_54/BiasAdd/ReadVariableOp¢"lstm_cell_54/MatMul/ReadVariableOp¢$lstm_cell_54/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_54/MatMul/ReadVariableOpReadVariableOp+lstm_cell_54_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_54/MatMulMatMulstrided_slice_2:output:0*lstm_cell_54/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_54/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_54_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0
lstm_cell_54/MatMul_1MatMulzeros:output:0,lstm_cell_54/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_54/addAddV2lstm_cell_54/MatMul:product:0lstm_cell_54/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_54/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_54_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_54/BiasAddBiasAddlstm_cell_54/add:z:0+lstm_cell_54/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_54/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstm_cell_54/splitSplit%lstm_cell_54/split/split_dim:output:0lstm_cell_54/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitn
lstm_cell_54/SigmoidSigmoidlstm_cell_54/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
lstm_cell_54/Sigmoid_1Sigmoidlstm_cell_54/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
lstm_cell_54/mulMullstm_cell_54/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ h
lstm_cell_54/ReluRelulstm_cell_54/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_54/mul_1Mullstm_cell_54/Sigmoid:y:0lstm_cell_54/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ {
lstm_cell_54/add_1AddV2lstm_cell_54/mul:z:0lstm_cell_54/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
lstm_cell_54/Sigmoid_2Sigmoidlstm_cell_54/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ e
lstm_cell_54/Relu_1Relulstm_cell_54/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_54/mul_2Mullstm_cell_54/Sigmoid_2:y:0!lstm_cell_54/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_54_matmul_readvariableop_resource-lstm_cell_54_matmul_1_readvariableop_resource,lstm_cell_54_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_864098*
condR
while_cond_864097*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ À
NoOpNoOp$^lstm_cell_54/BiasAdd/ReadVariableOp#^lstm_cell_54/MatMul/ReadVariableOp%^lstm_cell_54/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_54/BiasAdd/ReadVariableOp#lstm_cell_54/BiasAdd/ReadVariableOp2H
"lstm_cell_54/MatMul/ReadVariableOp"lstm_cell_54/MatMul/ReadVariableOp2L
$lstm_cell_54/MatMul_1/ReadVariableOp$lstm_cell_54/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®
«
#model_12_salt_seq_while_cond_863086@
<model_12_salt_seq_while_model_12_salt_seq_while_loop_counterF
Bmodel_12_salt_seq_while_model_12_salt_seq_while_maximum_iterations'
#model_12_salt_seq_while_placeholder)
%model_12_salt_seq_while_placeholder_1)
%model_12_salt_seq_while_placeholder_2)
%model_12_salt_seq_while_placeholder_3B
>model_12_salt_seq_while_less_model_12_salt_seq_strided_slice_1X
Tmodel_12_salt_seq_while_model_12_salt_seq_while_cond_863086___redundant_placeholder0X
Tmodel_12_salt_seq_while_model_12_salt_seq_while_cond_863086___redundant_placeholder1X
Tmodel_12_salt_seq_while_model_12_salt_seq_while_cond_863086___redundant_placeholder2X
Tmodel_12_salt_seq_while_model_12_salt_seq_while_cond_863086___redundant_placeholder3$
 model_12_salt_seq_while_identity
ª
model_12/salt_seq/while/LessLess#model_12_salt_seq_while_placeholder>model_12_salt_seq_while_less_model_12_salt_seq_strided_slice_1*
T0*
_output_shapes
: o
 model_12/salt_seq/while/IdentityIdentity model_12/salt_seq/while/Less:z:0*
T0
*
_output_shapes
: "M
 model_12_salt_seq_while_identity)model_12/salt_seq/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
ï

Ø
)__inference_model_12_layer_call_fn_864249
	salt_data
quantity_data
unknown:	
	unknown_0:	 
	unknown_1:	
	unknown_2:	
	unknown_3:	 
	unknown_4:	
	unknown_5:@
	unknown_6:
identity¢StatefulPartitionedCallº
StatefulPartitionedCallStatefulPartitionedCall	salt_dataquantity_dataunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_model_12_layer_call_and_return_conditional_losses_864230o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	salt_data:ZV
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namequantity_data
Õ

H__inference_lstm_cell_55_layer_call_and_return_conditional_losses_863598

inputs

states
states_11
matmul_readvariableop_resource:	3
 matmul_1_readvariableop_resource:	 .
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ N
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_namestates
ÿ	
d
E__inference_qty_seq_2_layer_call_and_return_conditional_losses_866749

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seed[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ù"
ã
while_body_863803
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_55_863827_0:	.
while_lstm_cell_55_863829_0:	 *
while_lstm_cell_55_863831_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_55_863827:	,
while_lstm_cell_55_863829:	 (
while_lstm_cell_55_863831:	¢*while/lstm_cell_55/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0³
*while/lstm_cell_55/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_55_863827_0while_lstm_cell_55_863829_0while_lstm_cell_55_863831_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_55_layer_call_and_return_conditional_losses_863744Ü
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_55/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity3while/lstm_cell_55/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/Identity_5Identity3while/lstm_cell_55/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ y

while/NoOpNoOp+^while/lstm_cell_55/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_55_863827while_lstm_cell_55_863827_0"8
while_lstm_cell_55_863829while_lstm_cell_55_863829_0"8
while_lstm_cell_55_863831while_lstm_cell_55_863831_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2X
*while/lstm_cell_55/StatefulPartitionedCall*while/lstm_cell_55/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
8
Ð
while_body_865852
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_54_matmul_readvariableop_resource_0:	H
5while_lstm_cell_54_matmul_1_readvariableop_resource_0:	 C
4while_lstm_cell_54_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_54_matmul_readvariableop_resource:	F
3while_lstm_cell_54_matmul_1_readvariableop_resource:	 A
2while_lstm_cell_54_biasadd_readvariableop_resource:	¢)while/lstm_cell_54/BiasAdd/ReadVariableOp¢(while/lstm_cell_54/MatMul/ReadVariableOp¢*while/lstm_cell_54/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_54/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_54_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0º
while/lstm_cell_54/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_54/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
*while/lstm_cell_54/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_54_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0¡
while/lstm_cell_54/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_54/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_54/addAddV2#while/lstm_cell_54/MatMul:product:0%while/lstm_cell_54/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_54/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_54_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstm_cell_54/BiasAddBiasAddwhile/lstm_cell_54/add:z:01while/lstm_cell_54/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_54/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
while/lstm_cell_54/splitSplit+while/lstm_cell_54/split/split_dim:output:0#while/lstm_cell_54/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitz
while/lstm_cell_54/SigmoidSigmoid!while/lstm_cell_54/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
while/lstm_cell_54/Sigmoid_1Sigmoid!while/lstm_cell_54/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_54/mulMul while/lstm_cell_54/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
while/lstm_cell_54/ReluRelu!while/lstm_cell_54/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_54/mul_1Mulwhile/lstm_cell_54/Sigmoid:y:0%while/lstm_cell_54/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_54/add_1AddV2while/lstm_cell_54/mul:z:0while/lstm_cell_54/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
while/lstm_cell_54/Sigmoid_2Sigmoid!while/lstm_cell_54/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q
while/lstm_cell_54/Relu_1Reluwhile/lstm_cell_54/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_54/mul_2Mul while/lstm_cell_54/Sigmoid_2:y:0'while/lstm_cell_54/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Å
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_54/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒy
while/Identity_4Identitywhile/lstm_cell_54/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ y
while/Identity_5Identitywhile/lstm_cell_54/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ð

while/NoOpNoOp*^while/lstm_cell_54/BiasAdd/ReadVariableOp)^while/lstm_cell_54/MatMul/ReadVariableOp+^while/lstm_cell_54/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_54_biasadd_readvariableop_resource4while_lstm_cell_54_biasadd_readvariableop_resource_0"l
3while_lstm_cell_54_matmul_1_readvariableop_resource5while_lstm_cell_54_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_54_matmul_readvariableop_resource3while_lstm_cell_54_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2V
)while/lstm_cell_54/BiasAdd/ReadVariableOp)while/lstm_cell_54/BiasAdd/ReadVariableOp2T
(while/lstm_cell_54/MatMul/ReadVariableOp(while/lstm_cell_54/MatMul/ReadVariableOp2X
*while/lstm_cell_54/MatMul_1/ReadVariableOp*while/lstm_cell_54/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
8
Ð
while_body_865566
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_54_matmul_readvariableop_resource_0:	H
5while_lstm_cell_54_matmul_1_readvariableop_resource_0:	 C
4while_lstm_cell_54_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_54_matmul_readvariableop_resource:	F
3while_lstm_cell_54_matmul_1_readvariableop_resource:	 A
2while_lstm_cell_54_biasadd_readvariableop_resource:	¢)while/lstm_cell_54/BiasAdd/ReadVariableOp¢(while/lstm_cell_54/MatMul/ReadVariableOp¢*while/lstm_cell_54/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_54/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_54_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0º
while/lstm_cell_54/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_54/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
*while/lstm_cell_54/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_54_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0¡
while/lstm_cell_54/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_54/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_54/addAddV2#while/lstm_cell_54/MatMul:product:0%while/lstm_cell_54/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_54/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_54_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstm_cell_54/BiasAddBiasAddwhile/lstm_cell_54/add:z:01while/lstm_cell_54/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_54/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
while/lstm_cell_54/splitSplit+while/lstm_cell_54/split/split_dim:output:0#while/lstm_cell_54/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitz
while/lstm_cell_54/SigmoidSigmoid!while/lstm_cell_54/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
while/lstm_cell_54/Sigmoid_1Sigmoid!while/lstm_cell_54/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_54/mulMul while/lstm_cell_54/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
while/lstm_cell_54/ReluRelu!while/lstm_cell_54/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_54/mul_1Mulwhile/lstm_cell_54/Sigmoid:y:0%while/lstm_cell_54/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_54/add_1AddV2while/lstm_cell_54/mul:z:0while/lstm_cell_54/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
while/lstm_cell_54/Sigmoid_2Sigmoid!while/lstm_cell_54/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q
while/lstm_cell_54/Relu_1Reluwhile/lstm_cell_54/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_54/mul_2Mul while/lstm_cell_54/Sigmoid_2:y:0'while/lstm_cell_54/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Å
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_54/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒy
while/Identity_4Identitywhile/lstm_cell_54/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ y
while/Identity_5Identitywhile/lstm_cell_54/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ð

while/NoOpNoOp*^while/lstm_cell_54/BiasAdd/ReadVariableOp)^while/lstm_cell_54/MatMul/ReadVariableOp+^while/lstm_cell_54/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_54_biasadd_readvariableop_resource4while_lstm_cell_54_biasadd_readvariableop_resource_0"l
3while_lstm_cell_54_matmul_1_readvariableop_resource5while_lstm_cell_54_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_54_matmul_readvariableop_resource3while_lstm_cell_54_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2V
)while/lstm_cell_54/BiasAdd/ReadVariableOp)while/lstm_cell_54/BiasAdd/ReadVariableOp2T
(while/lstm_cell_54/MatMul/ReadVariableOp(while/lstm_cell_54/MatMul/ReadVariableOp2X
*while/lstm_cell_54/MatMul_1/ReadVariableOp*while/lstm_cell_54/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
µ
Ã
while_cond_865851
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_865851___redundant_placeholder04
0while_while_cond_865851___redundant_placeholder14
0while_while_cond_865851___redundant_placeholder24
0while_while_cond_865851___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
Ý

H__inference_lstm_cell_54_layer_call_and_return_conditional_losses_866879

inputs
states_0
states_11
matmul_readvariableop_resource:	3
 matmul_1_readvariableop_resource:	 .
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ N
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/1

¸
)__inference_salt_seq_layer_call_fn_865474
inputs_0
unknown:	
	unknown_0:	 
	unknown_1:	
identity¢StatefulPartitionedCallè
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_salt_seq_layer_call_and_return_conditional_losses_863331o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
8
Ð
while_body_865709
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_54_matmul_readvariableop_resource_0:	H
5while_lstm_cell_54_matmul_1_readvariableop_resource_0:	 C
4while_lstm_cell_54_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_54_matmul_readvariableop_resource:	F
3while_lstm_cell_54_matmul_1_readvariableop_resource:	 A
2while_lstm_cell_54_biasadd_readvariableop_resource:	¢)while/lstm_cell_54/BiasAdd/ReadVariableOp¢(while/lstm_cell_54/MatMul/ReadVariableOp¢*while/lstm_cell_54/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_54/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_54_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0º
while/lstm_cell_54/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_54/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
*while/lstm_cell_54/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_54_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0¡
while/lstm_cell_54/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_54/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_54/addAddV2#while/lstm_cell_54/MatMul:product:0%while/lstm_cell_54/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_54/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_54_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstm_cell_54/BiasAddBiasAddwhile/lstm_cell_54/add:z:01while/lstm_cell_54/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_54/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
while/lstm_cell_54/splitSplit+while/lstm_cell_54/split/split_dim:output:0#while/lstm_cell_54/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitz
while/lstm_cell_54/SigmoidSigmoid!while/lstm_cell_54/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
while/lstm_cell_54/Sigmoid_1Sigmoid!while/lstm_cell_54/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_54/mulMul while/lstm_cell_54/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
while/lstm_cell_54/ReluRelu!while/lstm_cell_54/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_54/mul_1Mulwhile/lstm_cell_54/Sigmoid:y:0%while/lstm_cell_54/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_54/add_1AddV2while/lstm_cell_54/mul:z:0while/lstm_cell_54/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
while/lstm_cell_54/Sigmoid_2Sigmoid!while/lstm_cell_54/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q
while/lstm_cell_54/Relu_1Reluwhile/lstm_cell_54/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_54/mul_2Mul while/lstm_cell_54/Sigmoid_2:y:0'while/lstm_cell_54/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Å
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_54/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒy
while/Identity_4Identitywhile/lstm_cell_54/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ y
while/Identity_5Identitywhile/lstm_cell_54/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ð

while/NoOpNoOp*^while/lstm_cell_54/BiasAdd/ReadVariableOp)^while/lstm_cell_54/MatMul/ReadVariableOp+^while/lstm_cell_54/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_54_biasadd_readvariableop_resource4while_lstm_cell_54_biasadd_readvariableop_resource_0"l
3while_lstm_cell_54_matmul_1_readvariableop_resource5while_lstm_cell_54_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_54_matmul_readvariableop_resource3while_lstm_cell_54_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2V
)while/lstm_cell_54/BiasAdd/ReadVariableOp)while/lstm_cell_54/BiasAdd/ReadVariableOp2T
(while/lstm_cell_54/MatMul/ReadVariableOp(while/lstm_cell_54/MatMul/ReadVariableOp2X
*while/lstm_cell_54/MatMul_1/ReadVariableOp*while/lstm_cell_54/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
µ
Ã
while_cond_865708
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_865708___redundant_placeholder04
0while_while_cond_865708___redundant_placeholder14
0while_while_cond_865708___redundant_placeholder24
0while_while_cond_865708___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:

·
(__inference_qty_seq_layer_call_fn_866101
inputs_0
unknown:	
	unknown_0:	 
	unknown_1:	
identity¢StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_qty_seq_layer_call_and_return_conditional_losses_863872o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
µ
Ã
while_cond_866610
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_866610___redundant_placeholder04
0while_while_cond_866610___redundant_placeholder14
0while_while_cond_866610___redundant_placeholder24
0while_while_cond_866610___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
ñ
c
*__inference_qty_seq_2_layer_call_fn_866732

inputs
identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_qty_seq_2_layer_call_and_return_conditional_losses_864286o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ø
c
E__inference_qty_seq_2_layer_call_and_return_conditional_losses_866737

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
µ
Ã
while_cond_866324
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_866324___redundant_placeholder04
0while_while_cond_866324___redundant_placeholder14
0while_while_cond_866324___redundant_placeholder24
0while_while_cond_866324___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
8
Ð
while_body_863948
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_55_matmul_readvariableop_resource_0:	H
5while_lstm_cell_55_matmul_1_readvariableop_resource_0:	 C
4while_lstm_cell_55_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_55_matmul_readvariableop_resource:	F
3while_lstm_cell_55_matmul_1_readvariableop_resource:	 A
2while_lstm_cell_55_biasadd_readvariableop_resource:	¢)while/lstm_cell_55/BiasAdd/ReadVariableOp¢(while/lstm_cell_55/MatMul/ReadVariableOp¢*while/lstm_cell_55/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_55/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_55_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0º
while/lstm_cell_55/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_55/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
*while/lstm_cell_55/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_55_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0¡
while/lstm_cell_55/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_55/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_55/addAddV2#while/lstm_cell_55/MatMul:product:0%while/lstm_cell_55/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_55/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_55_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstm_cell_55/BiasAddBiasAddwhile/lstm_cell_55/add:z:01while/lstm_cell_55/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_55/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
while/lstm_cell_55/splitSplit+while/lstm_cell_55/split/split_dim:output:0#while/lstm_cell_55/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitz
while/lstm_cell_55/SigmoidSigmoid!while/lstm_cell_55/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
while/lstm_cell_55/Sigmoid_1Sigmoid!while/lstm_cell_55/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_55/mulMul while/lstm_cell_55/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
while/lstm_cell_55/ReluRelu!while/lstm_cell_55/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_55/mul_1Mulwhile/lstm_cell_55/Sigmoid:y:0%while/lstm_cell_55/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_55/add_1AddV2while/lstm_cell_55/mul:z:0while/lstm_cell_55/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
while/lstm_cell_55/Sigmoid_2Sigmoid!while/lstm_cell_55/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q
while/lstm_cell_55/Relu_1Reluwhile/lstm_cell_55/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_55/mul_2Mul while/lstm_cell_55/Sigmoid_2:y:0'while/lstm_cell_55/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Å
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_55/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒy
while/Identity_4Identitywhile/lstm_cell_55/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ y
while/Identity_5Identitywhile/lstm_cell_55/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ð

while/NoOpNoOp*^while/lstm_cell_55/BiasAdd/ReadVariableOp)^while/lstm_cell_55/MatMul/ReadVariableOp+^while/lstm_cell_55/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_55_biasadd_readvariableop_resource4while_lstm_cell_55_biasadd_readvariableop_resource_0"l
3while_lstm_cell_55_matmul_1_readvariableop_resource5while_lstm_cell_55_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_55_matmul_readvariableop_resource3while_lstm_cell_55_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2V
)while/lstm_cell_55/BiasAdd/ReadVariableOp)while/lstm_cell_55/BiasAdd/ReadVariableOp2T
(while/lstm_cell_55/MatMul/ReadVariableOp(while/lstm_cell_55/MatMul/ReadVariableOp2X
*while/lstm_cell_55/MatMul_1/ReadVariableOp*while/lstm_cell_55/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: "ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*
serving_defaultí
K
quantity_data:
serving_default_quantity_data:0ÿÿÿÿÿÿÿÿÿ
C
	salt_data6
serving_default_salt_data:0ÿÿÿÿÿÿÿÿÿ=
	salt_pred0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:É

layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
Ú
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
Ú
cell

state_spec
	variables
trainable_variables
regularization_losses
 	keras_api
!_random_generator
"__call__
*#&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
¼
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(_random_generator
)__call__
**&call_and_return_all_conditional_losses"
_tf_keras_layer
¼
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/_random_generator
0__call__
*1&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses"
_tf_keras_layer
»

8kernel
9bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses"
_tf_keras_layer
ó
@iter

Abeta_1

Bbeta_2
	Cdecay
Dlearning_rate8m9mEmFmGmHmImJm8v9vEvFvGvHvIvJv"
	optimizer
X
E0
F1
G2
H3
I4
J5
86
97"
trackable_list_wrapper
X
E0
F1
G2
H3
I4
J5
86
97"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ò2ï
)__inference_model_12_layer_call_fn_864249
)__inference_model_12_layer_call_fn_864817
)__inference_model_12_layer_call_fn_864839
)__inference_model_12_layer_call_fn_864735À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Þ2Û
D__inference_model_12_layer_call_and_return_conditional_losses_865132
D__inference_model_12_layer_call_and_return_conditional_losses_865439
D__inference_model_12_layer_call_and_return_conditional_losses_864762
D__inference_model_12_layer_call_and_return_conditional_losses_864789À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÝBÚ
!__inference__wrapped_model_863181	salt_dataquantity_data"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
Pserving_default"
signature_map
ø
Q
state_size

Ekernel
Frecurrent_kernel
Gbias
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V_random_generator
W__call__
*X&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
E0
F1
G2"
trackable_list_wrapper
5
E0
F1
G2"
trackable_list_wrapper
 "
trackable_list_wrapper
¹

Ystates
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
)__inference_salt_seq_layer_call_fn_865474
)__inference_salt_seq_layer_call_fn_865485
)__inference_salt_seq_layer_call_fn_865496
)__inference_salt_seq_layer_call_fn_865507Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ó2ð
D__inference_salt_seq_layer_call_and_return_conditional_losses_865650
D__inference_salt_seq_layer_call_and_return_conditional_losses_865793
D__inference_salt_seq_layer_call_and_return_conditional_losses_865936
D__inference_salt_seq_layer_call_and_return_conditional_losses_866079Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ø
_
state_size

Hkernel
Irecurrent_kernel
Jbias
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d_random_generator
e__call__
*f&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
H0
I1
J2"
trackable_list_wrapper
5
H0
I1
J2"
trackable_list_wrapper
 "
trackable_list_wrapper
¹

gstates
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
	variables
trainable_variables
regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
(__inference_qty_seq_layer_call_fn_866090
(__inference_qty_seq_layer_call_fn_866101
(__inference_qty_seq_layer_call_fn_866112
(__inference_qty_seq_layer_call_fn_866123Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ï2ì
C__inference_qty_seq_layer_call_and_return_conditional_losses_866266
C__inference_qty_seq_layer_call_and_return_conditional_losses_866409
C__inference_qty_seq_layer_call_and_return_conditional_losses_866552
C__inference_qty_seq_layer_call_and_return_conditional_losses_866695Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
$	variables
%trainable_variables
&regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
+__inference_salt_seq_2_layer_call_fn_866700
+__inference_salt_seq_2_layer_call_fn_866705´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ê2Ç
F__inference_salt_seq_2_layer_call_and_return_conditional_losses_866710
F__inference_salt_seq_2_layer_call_and_return_conditional_losses_866722´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
+	variables
,trainable_variables
-regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
*__inference_qty_seq_2_layer_call_fn_866727
*__inference_qty_seq_2_layer_call_fn_866732´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
È2Å
E__inference_qty_seq_2_layer_call_and_return_conditional_losses_866737
E__inference_qty_seq_2_layer_call_and_return_conditional_losses_866749´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
Ò2Ï
(__inference_pattern_layer_call_fn_866755¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_pattern_layer_call_and_return_conditional_losses_866762¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
": @2salt_pred/kernel
:2salt_pred/bias
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
®
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_salt_pred_layer_call_fn_866771¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_salt_pred_layer_call_and_return_conditional_losses_866781¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
/:-	2salt_seq/lstm_cell_54/kernel
9:7	 2&salt_seq/lstm_cell_54/recurrent_kernel
):'2salt_seq/lstm_cell_54/bias
.:,	2qty_seq/lstm_cell_55/kernel
8:6	 2%qty_seq/lstm_cell_55/recurrent_kernel
(:&2qty_seq/lstm_cell_55/bias
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÚB×
$__inference_signature_wrapper_865463quantity_data	salt_data"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
5
E0
F1
G2"
trackable_list_wrapper
5
E0
F1
G2"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
R	variables
Strainable_variables
Tregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
¢2
-__inference_lstm_cell_54_layer_call_fn_866798
-__inference_lstm_cell_54_layer_call_fn_866815¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ø2Õ
H__inference_lstm_cell_54_layer_call_and_return_conditional_losses_866847
H__inference_lstm_cell_54_layer_call_and_return_conditional_losses_866879¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
H0
I1
J2"
trackable_list_wrapper
5
H0
I1
J2"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
`	variables
atrainable_variables
bregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
¢2
-__inference_lstm_cell_55_layer_call_fn_866896
-__inference_lstm_cell_55_layer_call_fn_866913¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ø2Õ
H__inference_lstm_cell_55_layer_call_and_return_conditional_losses_866945
H__inference_lstm_cell_55_layer_call_and_return_conditional_losses_866977¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

total

count
	variables
	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
':%@2Adam/salt_pred/kernel/m
!:2Adam/salt_pred/bias/m
4:2	2#Adam/salt_seq/lstm_cell_54/kernel/m
>:<	 2-Adam/salt_seq/lstm_cell_54/recurrent_kernel/m
.:,2!Adam/salt_seq/lstm_cell_54/bias/m
3:1	2"Adam/qty_seq/lstm_cell_55/kernel/m
=:;	 2,Adam/qty_seq/lstm_cell_55/recurrent_kernel/m
-:+2 Adam/qty_seq/lstm_cell_55/bias/m
':%@2Adam/salt_pred/kernel/v
!:2Adam/salt_pred/bias/v
4:2	2#Adam/salt_seq/lstm_cell_54/kernel/v
>:<	 2-Adam/salt_seq/lstm_cell_54/recurrent_kernel/v
.:,2!Adam/salt_seq/lstm_cell_54/bias/v
3:1	2"Adam/qty_seq/lstm_cell_55/kernel/v
=:;	 2,Adam/qty_seq/lstm_cell_55/recurrent_kernel/v
-:+2 Adam/qty_seq/lstm_cell_55/bias/vÑ
!__inference__wrapped_model_863181«HIJEFG89h¢e
^¢[
Y¢V
'$
	salt_dataÿÿÿÿÿÿÿÿÿ
+(
quantity_dataÿÿÿÿÿÿÿÿÿ
ª "5ª2
0
	salt_pred# 
	salt_predÿÿÿÿÿÿÿÿÿÊ
H__inference_lstm_cell_54_layer_call_and_return_conditional_losses_866847ýEFG¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ 
"
states/1ÿÿÿÿÿÿÿÿÿ 
p 
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿ 
EB

0/1/0ÿÿÿÿÿÿÿÿÿ 

0/1/1ÿÿÿÿÿÿÿÿÿ 
 Ê
H__inference_lstm_cell_54_layer_call_and_return_conditional_losses_866879ýEFG¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ 
"
states/1ÿÿÿÿÿÿÿÿÿ 
p
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿ 
EB

0/1/0ÿÿÿÿÿÿÿÿÿ 

0/1/1ÿÿÿÿÿÿÿÿÿ 
 
-__inference_lstm_cell_54_layer_call_fn_866798íEFG¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ 
"
states/1ÿÿÿÿÿÿÿÿÿ 
p 
ª "c¢`

0ÿÿÿÿÿÿÿÿÿ 
A>

1/0ÿÿÿÿÿÿÿÿÿ 

1/1ÿÿÿÿÿÿÿÿÿ 
-__inference_lstm_cell_54_layer_call_fn_866815íEFG¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ 
"
states/1ÿÿÿÿÿÿÿÿÿ 
p
ª "c¢`

0ÿÿÿÿÿÿÿÿÿ 
A>

1/0ÿÿÿÿÿÿÿÿÿ 

1/1ÿÿÿÿÿÿÿÿÿ Ê
H__inference_lstm_cell_55_layer_call_and_return_conditional_losses_866945ýHIJ¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ 
"
states/1ÿÿÿÿÿÿÿÿÿ 
p 
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿ 
EB

0/1/0ÿÿÿÿÿÿÿÿÿ 

0/1/1ÿÿÿÿÿÿÿÿÿ 
 Ê
H__inference_lstm_cell_55_layer_call_and_return_conditional_losses_866977ýHIJ¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ 
"
states/1ÿÿÿÿÿÿÿÿÿ 
p
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿ 
EB

0/1/0ÿÿÿÿÿÿÿÿÿ 

0/1/1ÿÿÿÿÿÿÿÿÿ 
 
-__inference_lstm_cell_55_layer_call_fn_866896íHIJ¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ 
"
states/1ÿÿÿÿÿÿÿÿÿ 
p 
ª "c¢`

0ÿÿÿÿÿÿÿÿÿ 
A>

1/0ÿÿÿÿÿÿÿÿÿ 

1/1ÿÿÿÿÿÿÿÿÿ 
-__inference_lstm_cell_55_layer_call_fn_866913íHIJ¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ 
"
states/1ÿÿÿÿÿÿÿÿÿ 
p
ª "c¢`

0ÿÿÿÿÿÿÿÿÿ 
A>

1/0ÿÿÿÿÿÿÿÿÿ 

1/1ÿÿÿÿÿÿÿÿÿ ì
D__inference_model_12_layer_call_and_return_conditional_losses_864762£HIJEFG89p¢m
f¢c
Y¢V
'$
	salt_dataÿÿÿÿÿÿÿÿÿ
+(
quantity_dataÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ì
D__inference_model_12_layer_call_and_return_conditional_losses_864789£HIJEFG89p¢m
f¢c
Y¢V
'$
	salt_dataÿÿÿÿÿÿÿÿÿ
+(
quantity_dataÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 æ
D__inference_model_12_layer_call_and_return_conditional_losses_865132HIJEFG89j¢g
`¢]
S¢P
&#
inputs/0ÿÿÿÿÿÿÿÿÿ
&#
inputs/1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 æ
D__inference_model_12_layer_call_and_return_conditional_losses_865439HIJEFG89j¢g
`¢]
S¢P
&#
inputs/0ÿÿÿÿÿÿÿÿÿ
&#
inputs/1ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ä
)__inference_model_12_layer_call_fn_864249HIJEFG89p¢m
f¢c
Y¢V
'$
	salt_dataÿÿÿÿÿÿÿÿÿ
+(
quantity_dataÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿÄ
)__inference_model_12_layer_call_fn_864735HIJEFG89p¢m
f¢c
Y¢V
'$
	salt_dataÿÿÿÿÿÿÿÿÿ
+(
quantity_dataÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ¾
)__inference_model_12_layer_call_fn_864817HIJEFG89j¢g
`¢]
S¢P
&#
inputs/0ÿÿÿÿÿÿÿÿÿ
&#
inputs/1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¾
)__inference_model_12_layer_call_fn_864839HIJEFG89j¢g
`¢]
S¢P
&#
inputs/0ÿÿÿÿÿÿÿÿÿ
&#
inputs/1ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿË
C__inference_pattern_layer_call_and_return_conditional_losses_866762Z¢W
P¢M
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ 
"
inputs/1ÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ¢
(__inference_pattern_layer_call_fn_866755vZ¢W
P¢M
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ 
"
inputs/1ÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ@¥
E__inference_qty_seq_2_layer_call_and_return_conditional_losses_866737\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 ¥
E__inference_qty_seq_2_layer_call_and_return_conditional_losses_866749\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ 
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 }
*__inference_qty_seq_2_layer_call_fn_866727O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª "ÿÿÿÿÿÿÿÿÿ }
*__inference_qty_seq_2_layer_call_fn_866732O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ 
p
ª "ÿÿÿÿÿÿÿÿÿ Ä
C__inference_qty_seq_layer_call_and_return_conditional_losses_866266}HIJO¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 Ä
C__inference_qty_seq_layer_call_and_return_conditional_losses_866409}HIJO¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 ´
C__inference_qty_seq_layer_call_and_return_conditional_losses_866552mHIJ?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 ´
C__inference_qty_seq_layer_call_and_return_conditional_losses_866695mHIJ?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 
(__inference_qty_seq_layer_call_fn_866090pHIJO¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ 
(__inference_qty_seq_layer_call_fn_866101pHIJO¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ 
(__inference_qty_seq_layer_call_fn_866112`HIJ?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ 
(__inference_qty_seq_layer_call_fn_866123`HIJ?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ ¥
E__inference_salt_pred_layer_call_and_return_conditional_losses_866781\89/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
*__inference_salt_pred_layer_call_fn_866771O89/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ¦
F__inference_salt_seq_2_layer_call_and_return_conditional_losses_866710\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 ¦
F__inference_salt_seq_2_layer_call_and_return_conditional_losses_866722\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ 
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 ~
+__inference_salt_seq_2_layer_call_fn_866700O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª "ÿÿÿÿÿÿÿÿÿ ~
+__inference_salt_seq_2_layer_call_fn_866705O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ 
p
ª "ÿÿÿÿÿÿÿÿÿ Å
D__inference_salt_seq_layer_call_and_return_conditional_losses_865650}EFGO¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 Å
D__inference_salt_seq_layer_call_and_return_conditional_losses_865793}EFGO¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 µ
D__inference_salt_seq_layer_call_and_return_conditional_losses_865936mEFG?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 µ
D__inference_salt_seq_layer_call_and_return_conditional_losses_866079mEFG?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 
)__inference_salt_seq_layer_call_fn_865474pEFGO¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ 
)__inference_salt_seq_layer_call_fn_865485pEFGO¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ 
)__inference_salt_seq_layer_call_fn_865496`EFG?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ 
)__inference_salt_seq_layer_call_fn_865507`EFG?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ î
$__inference_signature_wrapper_865463ÅHIJEFG89¢~
¢ 
wªt
<
quantity_data+(
quantity_dataÿÿÿÿÿÿÿÿÿ
4
	salt_data'$
	salt_dataÿÿÿÿÿÿÿÿÿ"5ª2
0
	salt_pred# 
	salt_predÿÿÿÿÿÿÿÿÿ