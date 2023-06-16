from setuptools import setup

package_name = 'dabai_client'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ros',
    maintainer_email='ros@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "nonadet_client = dabai_client.nanodet_client:main",
            "dabai_test = dabai_client.dabai_test:main",
            "dabai_detect = dabai_client.nanodet_detect:main",
        ],
    },
)
